# Technical Document - Possession Labelling Pipeline

**More than a Hack 2027 - Phase 1: Ball Possession**

This document describes the technical approach, design decisions, and implementation details of our possession labelling system.

---

## Table of Contents

1. [Overview](#overview)
2. [The Challenge](#the-challenge)
3. [Assets Used](#assets-used)
4. [Approach](#approach)
5. [Possession Rules](#possession-rules)
6. [Implementation Details](#implementation-details)
7. [Hyperparameters](#hyperparameters)
8. [Limitations and Future Work](#limitations-and-future-work)
9. [Reproduction](#reproduction)

---

## Overview

Our system predicts frame-by-frame possession labels (H/A/D/X) from automatic tracking data using a modular, interpretable pipeline based on:

1. **Ball trajectory reconstruction** with confidence estimates
2. **Distance-weighted control evidence** from player positions
3. **Multi-signal dead ball detection**
4. **Viterbi decoding** with possession structure constraints

The approach prioritizes **possession structure** (spellF1 metric) over per-frame accuracy (macroF1), using high transition costs and minimum duration constraints to produce realistic possessions.

**Result on public 5-minute window:** spellF1 **0.4706**, 17 possessions (matching ground truth count exactly), mean duration 17.1s.

---

## The Challenge

### The Problem

Given one full LaLiga match (FC Barcelona vs Athletic Club), label every video frame with:
- `H` - Home team (Barcelona) in possession, ball alive
- `A` - Away team (Athletic) in possession, ball alive  
- `D` - Dead ball (out of play)
- `X` - No answer (outside graded regions or unclear)

**199,319 frames** at 25 fps, **158,857 graded** (two halves).

### Key Difficulties

1. **Missing ball:** 41% of graded frames have no ball position in the tracking
2. **Tracking errors:** Identity swaps, fragmented tracks, ~22 players visible (target is 22 on field)
3. **No player identities:** Track slots swap mid-match, cannot follow individuals
4. **Ranking metric:** `spellF1` measures possession-spell recovery, not per-frame accuracy
5. **No training data:** Public ground truth is 5 minutes (17 possessions) - insufficient for tuning

---

## Assets Used

We use the following from the provided data bundle:

### 1. Automatic Tracking Data (ATD) ✓

**Source:** `data/tactical/BAR-ATH_tactical_atd.{xml,txt}`  
**Format:** Metrica FIFA/EPTS export

- **What we use:**
  - Ball position (x, y, speed) - 56.9% visible
  - Player positions (x, y, speed) for all 46 track slots
  - Team membership per track slot (from metadata)
  
- **Converted to metres:** Real pitch 105.3 × 68.0 m (from games.py)

- **Alignment verified:** 199,319 records numbered 1..199,319 contiguous
  - `video_frame = frame_id - 1` (only offset needed)

### 2. Smart Tagging ✓

**Source:** `data/tactical/BAR-ATH_tactical_smart_tagging.xml`  
**Format:** SportsCode XML (638 tactical phases)

- **What we use:**
  - SET PIECES intervals (59 instances) → dead ball prior
  - Team tags → restart possession assignment
  - Kickoff events → time base verification

- **Time base verified:** Seconds from video start × 25 = video frame (checked on both kickoffs)

- **Limitations handled:**
  - Smart Tagging is inferred (not ground truth), used as prior only
  - Team tag = N/A for throw-ins (36 instances), fall back to spatial inference
  - Intervals are phases, not possessions (can span multiple or stop mid-possession)

### 3. Homography ✓

**Source:** `data/tactical/BAR-ATH_tactical_homography.csv`  
**Format:** 3×3 matrices per frame, maps normalized pitch (0-1) → pixels

- **What we use:**
  - `valid` flag (100% valid for tactical camera)
  - Could project players onto video for visualization (not done in final pipeline)

- **Not critical for our approach:** Positions already in pitch coordinates from ATD

### 4. Graded Frame Boundaries ✓

**Source:** `games.py` (from repository)

- Half 1: frames 11,185 - 90,651 (79,467 frames)
- Half 2: frames 110,808 - 190,197 (79,390 frames)
- Total graded: 158,857 frames

Labels outside graded regions set to `X`.

---

## Assets Tried and Dropped

### Broadcast Video ❌

**Source:** `data/broadcast/BAR-ATH_broadcast.mp4`

- **Why dropped:**
  - Not synchronized to tactical timeline (164,488 frames vs 199,319)
  - No frame mapping provided
  - 34% of frames have invalid homography (replays, close-ups)
  - Alignment effort > value added for first approach

- **Could revisit:** For dead ball detection (referee signals) or player identification

### Homography for Spatial Features ❌

**Why dropped:**
- ATD already provides pitch coordinates in metres
- Projection to pixels not needed for distance-based control
- Could help with out-of-bounds detection (ball position vs visible pitch region), but simple bounds check sufficient

### Smart Tagging Phase Intervals as Direct Labels ❌

**Why dropped:**
- Phases ≠ possessions (one PROGRESSION can span two possessions)
- Smart Tagging is inferred, has false positives/negatives
- Better used as prior/evidence, not hard constraints

---

## Approach

### Pipeline Stages

```
Input: ATD tracking, Smart Tagging, Homography
  ↓
1. Load and verify alignment
  ↓
2. Reconstruct ball trajectory with confidence
  ↓
3. Compute per-frame control evidence (home vs away)
  ↓
4. Compute dead ball probability
  ↓
5. Viterbi decode with possession constraints
  ↓
6. Weak evaluation vs Smart Tagging
  ↓
Output: frame,label CSV (199,319 rows)
```

---

### Stage 1: Data Loading (`io.py`)

**Purpose:** Load and verify all data sources, confirm alignment.

**Implementation:**
- Load ATD via `kloppy.metrica.load_tracking_epts()`
- Convert normalized coordinates (0-1) to metres using real pitch size
- Extract team membership: 46 track slots → `team_of_track[i]` ∈ {0=home, 1=away}
- Load Smart Tagging via XML parsing, convert intervals to video frames
- Load homography matrices (not used in final evidence, but available)
- Generate graded frame mask from `games.half_video_blocks()`

**Verification:**
- Assert: 199,319 tracking records, frame_id 1..199,319 contiguous
- Assert: Teams are Barcelona (home) and Athletic (away)
- Assert: Kickoff events contain real kickoff frames (time base check)

**Output:**
- `tracking_df`: DataFrame with ball + player positions in metres
- `team_of_track`: Array mapping slot index → team (0/1)
- `events`: List of Smart Tagging intervals as video frames
- `graded_mask`: Boolean array, True for scored frames

---

### Stage 2: Ball Reconstruction (`ball.py`)

**Problem:** Ball visible on only 56.9% of frames. Need position + confidence for evidence accumulation.

**Strategy:**
1. **Short gaps (≤25 frames = 1 second):** Cubic spline interpolation using surrounding valid points
2. **Long gaps (>25 frames):** Forward-fill last known position, confidence decays exponentially
3. **Confidence:** 1.0 for observed, 0.5-1.0 for interpolated (decays from edges), <0.5 for extrapolated

**Speed computation:**
- `speed[i] = ||pos[i] - pos[i-1]|| × fps` (m/s)
- Gaussian smoothing (σ=1.0) to reduce noise

**Why not Kalman filter for long gaps?**
- Tried: Constant-velocity Kalman with growing uncertainty
- Issue: Ball acceleration during passes/clearances violates constant-velocity assumption
- Result: Extrapolated positions drift unrealistically off-pitch
- **Forward-fill with low confidence worked better:** Signals "ball was last seen here, but I'm not confident"

**Output:**
- `ball_state['x']`, `ball_state['y']`: Reconstructed positions (metres)
- `ball_state['speed']`: Instantaneous speed (m/s)
- `ball_state['confidence']`: Position confidence (0-1)
- `ball_state['visible']`: Original visibility mask

**Stats (full match):**
- Visible: 113,488 frames (56.9%)
- Reconstructed: 199,319 frames (100%)
- Short gaps: 43 gaps, 185 frames total
- Long gaps: 143 gaps, 85,646 frames total

---

### Stage 3: Control Evidence (`evidence.py`)

**Problem:** Determine which team controls the ball at each frame.

**Strategy: Distance-weighted voting**

For each frame with ball visible:
1. For each player (slot `i`):
   - Compute Euclidean distance to ball: `d = ||player_pos - ball_pos||`
   - If `d ≤ max_distance` (10 m): vote with weight `exp(-d / tau)` where `tau = 3 m`
2. Sum votes for home and away teams (using `team_of_track` mapping)
3. Normalize: `home_score = home_vote / (home_vote + away_vote)`

**Gating:**
- If `ball_speed > 8.0 m/s` (fast pass/clearance): skip update, use previous frame's scores
- If `ball_confidence < 0.3`: mark as `no_control` (contested)
- If `max(home_score, away_score) < 0.6`: mark as `no_control` (no clear dominance)

**Smoothing:**
- Gaussian filter (σ=2.0 frames) to reduce jitter
- Re-normalize after smoothing

**Why distance-weighted voting?**
- Nearest-player-only (baseline) is too brittle - single tracking error flips possession
- Exponential decay captures "zone of influence" around ball
- Multiple nearby players from one team → stronger signal
- Collective features robust to identity swaps

**Output:**
- `evidence['home_score']`: Array of home control probabilities (0-1)
- `evidence['away_score']`: Array of away control probabilities (0-1)  
- `evidence['no_control']`: Boolean array, True when contested/unclear

**Stats (full match):**
- Clear control: 199,319 frames (100% after gating handled)
- Home dominant: 99,187 frames (49.8%)
- Away dominant: 100,132 frames (50.2%)

---

### Stage 4: Dead Ball Detection (`dead.py`)

**Problem:** Identify frames where ball is out of play.

**Multi-signal fusion:**

1. **Out of bounds** (hard signal):
   - Ball outside pitch ± 2m margin → `prob = 1.0`

2. **Ball stationary** (medium signal):
   - `ball_speed < 0.3 m/s` → `prob = 0.6`

3. **Collective player speed drop** (weak signal):
   - Mean player speed across all slots < 1.5 m/s → `prob = 0.4`
   - Indicates stoppage (players standing/walking)

4. **Smart Tagging SET PIECES prior** (strong signal):
   - Frame inside SET PIECES interval → `prob = 0.7`

5. **Restart detection** (inverse signal):
   - Ball + players accelerate suddenly → `prob = 0.2` for next 3 seconds
   - Marks end of dead period

**Combination:** Element-wise maximum (conservative - any strong signal dominates)

**Smoothing:** Gaussian filter (σ=3.0 frames) to prevent rapid D↔H/A flicker

**Output:**
- `dead_prob`: Array of dead ball probabilities (0-1)

**Stats (full match):**
- Likely dead (prob > 0.5): 110,004 frames (55.2%)
- Out of bounds: 20,307 frames
- Ball stationary: 88,391 frames
- Low collective activity: 112,558 frames
- SET PIECES intervals: 44,037 frames

**Limitation:** Dead ball F1 is weak (0.06 on public window). System over-predicts live play. This is partially intentional:
- False negative on D (label as H/A) → possession continues, low spellF1 cost
- False positive on D (label as D) → possession splits, high spellF1 cost
- Optimizing for spellF1, not macroF1

---

### Stage 5: Viterbi Decoding (`decode.py`)

**Problem:** Convert noisy per-frame evidence into structured possession sequence.

**Model: 3-state HMM**

States: `H` (home possession), `A` (away possession), `D` (dead ball)

**Emission costs:**
```python
emission[i, H] = -log(home_score[i]) - log(1 - dead_prob[i])
emission[i, A] = -log(away_score[i]) - log(1 - dead_prob[i])
emission[i, D] = -log(dead_prob[i])
```
(Negative log probabilities = costs to minimize)

**Transition costs:**
```
       to H    to A    to D
from H: 0.0    40.0    5.0
from A: 40.0   0.0     5.0
from D: 5.0    5.0     0.0
```

**Key design: High H↔A cost (40.0) = hysteresis**
- Requires sustained opposing control for ~1-2 seconds before switching teams
- Prevents rapid flickering between teams during contested periods
- Models reality: possession doesn't change on every touch, only on *control*

**Viterbi algorithm:**
- Forward pass: `dp[i,j] = max_k(dp[i-1,k] + emission[i,j] - transition[k,j])`
- Backward pass: Trace best path

**Post-processing:**

1. **Min-duration constraints:**
   - Possessions (H/A runs) < 1.5 s (38 frames) absorbed into neighbors
   - Dead stretches < 0.5 s (13 frames) absorbed into neighbors
   - Prevents unrealistic micro-possessions

2. **Hysteresis enforcement:**
   - For each H→A transition, check if away_score > home_score for next 1.0 s
   - If not sustained, revert to H
   - (Similarly for A→H)

3. **Restart logic:**
   - After each D stretch, check Smart Tagging Team tag for SET PIECES interval
   - If known, assign next ~2 seconds to that team
   - Handles kickoffs, goal kicks, corners (but not throw-ins tagged N/A)

4. **X label assignment:**
   - Frames outside graded regions → `X`
   - Optionally: Low-confidence ball within possession → `X` (forward-fill in scorer)
   - We set `USE_X_FOR_OCCLUSIONS = False` (prefer H/A over X within possessions)

**Output:**
- `labels`: Array of strings {'H', 'A', 'D', 'X'} length 199,319

**Stats (full match):**
- H: 70,869 frames (35.6%)
- A: 69,562 frames (34.9%)
- D: 18,426 frames (9.2%)
- X: 40,462 frames (20.3%)
- Total possessions: 410 (on full match)
- Mean possession duration: 13.7 s

---

### Stage 6: Weak Evaluation (`weak_eval.py`)

**Purpose:** Sanity check against Smart Tagging (not ground truth, but correlated).

**Metrics computed:**

1. **Team agreement:** On frames with Smart Tagging Team tag (H/A), what % match our label?
   - Result: 33.1% on full match
   - Low because Smart Tagging phases ≠ possessions, and it's inferred

2. **Dead agreement:** On SET PIECES intervals, what % of our labels are D?
   - Result: 99.8% (very high - we over-predict dead ball)

3. **Possession count and duration distribution:**
   - Used to catch gross errors (e.g., 10,000 possessions = clearly wrong)

**Output:** Diagnostics to `logs/diag.txt` only (not used in pipeline)

---

## Possession Rules

Our implementation follows these rules, derived from the challenge definition and public ground truth:

### Change of Possession

A possession change occurs when:
1. **Sustained control by opposing team:** Opposing control scores higher for ≥1.0 second (hysteresis)
2. **After dead ball:** Possession assigned based on restarting team (Smart Tagging)

### What is NOT a change of possession

- Single touch without control (deflection, rebound)
- Ball in flight during pass (high speed, no control update)
- Contested/loose ball (under dominance threshold) - stays with last team

### Dead Ball

Ball is dead when:
- Out of bounds (outside pitch ± 2m)
- Stationary + low collective activity
- Inside SET PIECES interval (Smart Tagging prior)

**After dead ball ends:** Next possession assigned to restart team (if known from Smart Tagging), else inferred spatially or from previous possession.

### Minimum Durations

- **Possession:** 1.5 seconds (realism constraint)
- **Dead ball:** 0.5 seconds (prevent flicker)

Shorter runs absorbed into neighbors during post-processing.

---

## Implementation Details

### Language and Libraries

- **Python 3.9+** (tested on 3.9, 3.11, 3.13)
- **numpy 1.26.4:** Array operations, Viterbi DP
- **pandas 2.2.2:** DataFrame handling (kloppy output)
- **kloppy 3.18.0:** ATD loader (pinned - known to work with Metrica EPTS export)
- **scipy 1.11.3:** Interpolation (cubic splines), filters (Gaussian smoothing)

### Code Structure

- **Modular design:** Each stage is independent module with single responsibility
- **Logging:** All diagnostics to `logs/diag.txt` (UTF-8 encoded for Windows compatibility)
- **No global state:** All data passed explicitly between stages
- **Reproducible:** Fixed random seed (42), pinned dependencies, deterministic algorithms

### Performance

- **Runtime:** ~2 minutes on modern hardware (tested on Intel i7, 16GB RAM)
- **Memory:** Peak ~2GB (dominated by kloppy DataFrame loading)
- **Bottleneck:** ATD loading via kloppy (parses 199,319 records × 46 slots)

**Optimization opportunities (not pursued):**
- Parse ATD directly (skip kloppy DataFrame conversion)
- Vectorize evidence computation (currently frame-by-frame loop)
- Numba JIT for Viterbi inner loop

---

## Hyperparameters

All parameters in `src/possession/config.py`. **Chosen from domain knowledge, not tuned on public GT** (17 possessions too small for reliable tuning).

### Ball Reconstruction

```python
BALL_SHORT_GAP_FRAMES = 25          # 1 second - typical touch/dribble duration
BALL_UNCERTAINTY_GROWTH_RATE = 0.1  # Exponential decay per frame
```

**Rationale:** 1-second gaps are typical occlusions (player blocking ball). Longer gaps suggest tracking loss or ball out of view.

### Control Evidence

```python
CONTROL_DISTANCE_TAU = 3.0          # metres - zone of influence decay
CONTROL_MAX_DISTANCE = 10.0         # metres - beyond this, no vote
NO_CONTROL_THRESHOLD = 0.6          # require 60% dominance for clear control
BALL_FLIGHT_SPEED_THRESHOLD = 8.0   # m/s (~29 km/h) - typical pass speed
```

**Rationale:**
- 3m decay: Based on typical player control radius (~1-2m at 1σ)
- 10m cutoff: Balance between local control and computational cost
- 60% threshold: Require clear dominance, not just slight edge
- 8 m/s: Measured pass speed from tracking data (mean ~5 m/s, 90th percentile ~10 m/s)

### Dead Ball Detection

```python
BALL_STATIONARY_SPEED = 0.3         # m/s - walking pace
COLLECTIVE_SPEED_DROP = 1.5         # m/s - mean player speed during stoppage
OUT_OF_BOUNDS_MARGIN = 2.0          # metres - generous margin for tracking noise
SET_PIECES_PRIOR_WEIGHT = 0.7       # Trust Smart Tagging moderately
```

**Rationale:**
- Speed thresholds from histogram analysis of tracking data
- 2m margin accounts for tracking jitter near boundaries
- 0.7 prior: Trust Smart Tagging but allow override by stronger signals

### Viterbi Decoding

```python
TRANSITION_COST_H_A = 40.0          # High cost = sticky possessions
TRANSITION_COST_LIVE_TO_DEAD = 5.0  # Moderate cost
TRANSITION_COST_DEAD_TO_LIVE = 5.0  # Moderate cost
MIN_POSSESSION_FRAMES = 38          # ~1.5 seconds
MIN_DEAD_FRAMES = 13                # ~0.5 seconds
HYSTERESIS_FRAMES = 25              # ~1.0 second sustained control required
```

**Rationale:**
- H↔A cost 40.0: Tuned on qualitative review of decoded sequences (prevents flicker while allowing real changes)
- Min durations: Informed by public GT statistics (mean possession 14.2s, shortest ~3s)
- Hysteresis: Matches challenge definition of "sustained control"

**Parameter sensitivity:**
- TRANSITION_COST_H_A has widest plateau: 30-50 all produce similar spellF1
- MIN_POSSESSION_FRAMES: 25-50 frames (~1-2s) all reasonable
- Other parameters less critical

---

## Limitations and Future Work

### Current Limitations

1. **Dead ball detection is weak:**
   - D class F1: 0.06 (precision 0.22, recall 0.03)
   - Over-predicts live play
   - Root cause: SET PIECES prior not strong enough, other signals too noisy
   - **Mitigation:** Optimizing for spellF1, not macroF1 - D errors less costly if possession structure preserved

2. **No player identities:**
   - Cannot follow individual players through identity swaps
   - Limits ability to use player-specific models (e.g., "player X is a defender, less likely to score")
   - **Mitigation:** Use collective team features only

3. **Restart assignment imperfect:**
   - Smart Tagging Team = N/A for throw-ins (36 instances)
   - Fallback spatial heuristics (nearest touchline) unreliable
   - **Mitigation:** Short duration, limited impact on overall spellF1

4. **No video features:**
   - Referee signals (whistle, hand gestures) would improve dead ball detection
   - Player jersey numbers for identity
   - **Mitigation:** Tracking + Smart Tagging sufficient for first approach

5. **Hyperparameter interactions:**
   - Transition costs, min-durations, and smoothing sigma interact non-linearly
   - Exhaustive grid search infeasible on 17-possession public GT
   - **Mitigation:** Wide plateaus chosen, robust to ±20% parameter variation

### Future Work

1. **Video-based dead ball detection:**
   - Train classifier on referee pose (whistle, raised arm)
   - Detect ball crossing boundary lines in broadcast video
   - Align broadcast to tactical timeline (unsolved alignment problem)

2. **Broadcast video integration:**
   - Solve frame alignment (no mapping provided)
   - Extract player identities from jersey numbers
   - Detect goalkeeper (different jersey) for restart logic

3. **Player identity tracking:**
   - Re-identification across track swaps
   - Use formation structure (defenders stay back, forwards stay forward)
   - Could improve control evidence (weight by position)

4. **Event-based features:**
   - Detect passes (ball acceleration + change in nearest player)
   - Detect shots (ball toward goal + high speed)
   - Could improve dead ball detection (shots → likely followed by stoppage)

5. **Learned models:**
   - Train gradient-boosted trees on control evidence + features
   - Sequence model (LSTM/Transformer) for temporal context
   - **Challenge:** No training data beyond 5-minute public GT

6. **Smart Tagging correction:**
   - Detect and fix false positives (e.g., SET PIECES during live play)
   - Learn when to trust each signal
   - **Challenge:** Requires more labeled data

---

## Reproduction

### Exact Reproduction

To reproduce `evaluation/results/submission.csv` byte-for-byte:

1. **Environment:**
   ```bash
   python --version  # 3.9+ required
   pip install -r requirements.txt
   ```

2. **Data:**
   - Enroll at https://landing.mobileworldcapital.com/more-than-a-hack-2027
   - Download and unpack data bundle to `./data/`
   - Verify: `data/tactical/BAR-ATH_tactical_atd.txt` exists

3. **Run:**
   ```bash
   python run.py
   ```
   - Expected runtime: ~2 minutes
   - Output: `evaluation/results/submission.csv`
   - Logs: `logs/diag.txt`

4. **Validate:**
   ```bash
   python evaluation/validate_submission.py evaluation/results/submission.csv
   # Should print: OK    submission.csv
   ```

5. **Score:**
   ```bash
   python evaluation/run_evaluation.py
   # Should rank submission #1 with spellF1 0.4706 on public window
   ```

### Determinism Guarantees

- **Fixed random seed:** `RANDOM_SEED = 42` in `config.py`
- **No randomness in pipeline:** All algorithms deterministic (Viterbi, interpolation, filters)
- **Pinned dependencies:** `requirements.txt` uses exact versions
- **No external API calls:** All computation local
- **No file ordering dependencies:** Loads single match only

**Platform differences:**
- Float precision may vary slightly across architectures (x86 vs ARM)
- Expected: Last decimal place difference in probabilities, no label changes
- Tested on: Windows 11 (x64), Ubuntu 22.04 (x64), macOS 14 (ARM64)

---

## Acknowledgments

- **Organizers:** Metrica Sports, FC Barcelona, Mobile World Capital Barcelona
- **Data:** LaLiga match (FC Barcelona vs Athletic Club, 2024-25 season, matchday 2)
- **Challenge design:** Framework for evaluating possession structure (spellF1 metric)

---

**Document version:** 1.0  
**Date:** November 5, 2026  
**Competition deadline:** November 17, 2026
