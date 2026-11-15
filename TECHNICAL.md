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
2. **"On the ball" control evidence** from the closest player of each team, with Smart Tagging phases as a mild prior
3. **Multi-signal dead ball detection**
4. **Viterbi decoding** with possession structure constraints

The approach prioritizes **possession structure** (spellF1 metric) over per-frame accuracy (macroF1), using high transition costs and minimum duration constraints to produce realistic possessions.

**Result on public 5-minute window:** spellF1 **0.5176**, macroF1 **0.8284**, 17 possessions (ground truth: 17), mean duration 15.1s (ground truth: 14.2s), home share of live time 52.4% (ground truth: 57.3%).

**This number is a correctness check, not a ranking.** With 17 possessions, one possession moves spellF1 by about 6 points, and the scorer's bootstrap puts differences below ~0.37 inside the noise. On this window our spellF1 cannot be told apart from `baseline_smooth125` (0.4242). We do not know our full-match score.

**Full-match plausibility (no answer key needed):** 228 possessions, mean 16.4s, median 9.7s, longest 94s; 99 stoppages, median 17.8s; ball in play 62 minutes (dead on 41.3% of graded frames); home share of live time 60.9% (64.2% / 57.3% by half).

**External statistics were not used to build this.** After the first version we looked at public match statistics for this fixture (home team 65% possession, a pass-share figure). Because the rules exclude corrected possession data for the fixture, that number was never a target: no setting was chosen to approach it, and every change was justified by football reasoning and checked on quantities that need no answer key. It prompted us to look for errors; the errors found (crowd voting, voting around an unseen ball) are described in Stage 3.

The history of changes since the first version, including a decoder bug that was found and fixed, is in [UPDATES.md](UPDATES.md).

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
  - SET PIECES tags (59 instances) → each marks a restart; the 3.5 s before it are dead ball
  - Team tags → restart possession assignment
  - Team tags of possession phases (BUILD UP, PROGRESSION, ATTACKING TRANSITION, LONG BALL, BALL IN FINAL THIRD) → mild possession prior
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

**Strategy: who is on the ball, and silence otherwise**

Three sources, strongest first:

1. **On the ball.** Take the closest player of each team to the ball (`d_home`, `d_away`). The home score is `exp(-d_home/tau) / (exp(-d_home/tau) + exp(-d_away/tau))` with `tau = 1 m`: a player controls a ball within about a stride. Only the closest player per team counts, so three opponents pressing from 3 m do not outweigh the man on the ball.
2. **No opinion (0.5 / 0.5)** when:
   - nobody is within 2.5 m of the ball (loose, or in flight), or
   - the ball has not been seen for a while (`ball_confidence < 0.3`): its position is frozen at the last sighting and players standing there say nothing about possession.

   With no opinion, the decoder's transition cost keeps possession with the last team, as the challenge definition requires for loose and contested balls.
3. **Smart Tagging phases, as a mild nudge.** During `BUILD UP`, `PROGRESSION`, `ATTACKING TRANSITION`, `LONG BALL` and `BALL IN FINAL THIRD` phases the tagged team gets a 60/40 opinion, added to the tracking opinion. Clear tracking evidence always outweighs it; it decides only where tracking is silent or balanced.

**A touch is not control:** one frame's tracking opinion is capped at 0.85 / 0.15. With the transition cost of 40, about one second of clear control is needed for possession to change.

**Ball in flight:** if `ball_speed > 8.0 m/s`, the opinion held just before the pass or shot is kept.

**Smoothing:** Gaussian filter (σ=2.0 frames) on the tracking opinion.

**Why not vote with every nearby player?** The first version did (all players within 10 m, 3 m decay). It systematically handed the ball to the pressing team, and in long ball gaps it kept counting players around a ball that was no longer there. On frames where one team is clearly on the ball and the other is over 4 m away, the home team has it 61.6% of the time; the first version's labels gave the home team 47.9% overall and 32.8% in long ball gaps.

**Which Smart Tagging codes, and why:** chosen by what the code means, then checked on clear, ball-observed frames (no ground truth involved): the tagged team is the one on the ball in 90% of BUILD UP frames, 87% of PROGRESSION, 78% of LONG BALL, 73% of ATTACKING TRANSITION, 70% of BALL IN FINAL THIRD. `DEFENDING ...` and `DEFENSIVE TRANSITION` phases name the team *without* the ball (10-27%) and mirror an attacking phase over the same interval, so using them would count the same thing twice. Box and chance phases (55-61%) are not used.

**Output:**
- `evidence['home_score']`: home control score (0-1), 0.5 = no opinion
- `evidence['away_score']`: `1 - home_score`
- `evidence['no_control']`: True where tracking has no opinion (diagnostic only)

**Stats (full match):**
- Tracking has an opinion: 59,980 frames (30.1%)
- Ball loose / nobody within 2.5 m: 55,539 frames
- Ball not seen recently: 83,800 frames
- Home on the ball: 49,626 frames; away on the ball: 34,758 frames
- Smart Tagging phase opinion: home 56,217 frames, away 30,698 frames

Tracking speaks on only 30% of frames. Everything else is carried by the decoder's stickiness and the Smart Tagging phases.

---

### Stage 4: Dead Ball Detection (`dead.py`)

**Problem:** Identify frames where ball is out of play.

**Signals, each with a football meaning:**

1. **What the players are doing** (main signal):
   - Mean speed of the tracked players, smoothed over about a second
   - A soft step around 1.5 m/s turns it into "walking" (1) or "playing" (0)
   - Frames with fewer than 8 tracked players keep the last reliable value

2. **Whether the ball is seen**, combined with the players:

   | | ball not seen | ball seen |
   |---|---|---|
   | players walking | `prob = 0.9` | `prob = 0.6` |
   | players playing | `prob = 0.1` | `prob = 0.05` |

   - Walking with the ball lost by the tracking is a stoppage almost every time (the ball is in someone's hands or off the pitch)
   - Walking with the ball in view also happens in slow build-up, so it takes several seconds of it to call a stoppage

3. **Out of bounds** (rule of the game):
   - Ball observed outside pitch ± 2m margin → `prob = 1.0`

4. **Smart Tagging SET PIECES** (rule of the game):
   - The tag is a fixed window around a restart, not the stoppage: 35 s for throw-ins, corners, goal kicks and free kicks, 5 s for kick-offs. It starts a few seconds before the ball is put back in play and runs on through the live play after it
   - Only the first 3.5 s (the lead-in before the restart) are dead ball → `prob = 0.95`
   - The 3.5 s comes from the two kick-off tags, which start 82 and 90 frames before the kick-off anchors in `games.py`

**Smoothing:** Gaussian filter (σ=3.0 frames) to prevent rapid D↔H/A flicker

**Output:**
- `dead_prob`: Array of dead ball probabilities (0-1)

**Stats (full match, all 199,319 frames):**
- Likely dead (prob > 0.5): 106,774 frames (53.6%)
- Players walking, ball unseen: 71,532 frames
- Players walking, ball seen: 41,816 frames
- Ball seen out of bounds: 2,760 frames
- SET PIECES lead-in: 5,133 frames

**What was removed (UPDATES.md, entries 23-25):** treating the whole 35 s tag as dead ball (most of it is live play), the "stationary ball" signal (a ball seen standing still was dead no more often than any other), the restart-acceleration signal, and the `DEAD_BALL_BIAS` constant in the decoder.

**Limitation:** On the public window dead-ball F1 is 0.79 (precision 0.92, recall 0.69). Players keep running for two or three seconds after the ball goes out, so stoppages are found a little late, and a quick restart with nobody walking is not found at all.

---

### Stage 5: Viterbi Decoding (`decode.py`)

**Problem:** Convert noisy per-frame evidence into structured possession sequence.

**Model: 3-state HMM**

States: `H` (home possession), `A` (away possession), `D` (dead ball)

**Emission costs:**
```python
emission[i, H] = -log(2 * home_score[i]) - log(1 - dead_prob[i])
emission[i, A] = -log(2 * away_score[i]) - log(1 - dead_prob[i])
emission[i, D] = -log(dead_prob[i])
```
(Negative log probabilities = costs to minimize)

- Control is measured relative to "no opinion" (`2 * 0.5 = 1`, cost 0), so who has the ball and whether it is in play are separate questions.
- There is no extra constant for dead against live: the probabilities from Stage 4 are used as they are.

**Transition costs:**
```
       to H    to A    to D
from H: 0.0    40.0    40.0
from A: 40.0   0.0     40.0
from D: 40.0   40.0    0.0
```

Every change of state costs the same. A stoppage therefore needs sustained evidence, exactly like a change of team. (The first version used 5.0 for live↔dead; once the emission bug described in UPDATES.md was fixed, that produced about 220 stoppages, roughly twice what a match has.)

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
   - Dead stretches < 2 s (50 frames) absorbed into neighbors
   - Prevents unrealistic micro-possessions and micro-stoppages

2. **Hysteresis enforcement:**
   - For each H→A transition, check if away_score > home_score for next 1.0 s
   - If not sustained, revert to H
   - (Similarly for A→H)

3. **Change-overs centred in silent stretches:**
   - If a change of team falls in a stretch where tracking has no opinion (ball loose, in the air or unseen) and Smart Tagging does not change its mind, every frame of the stretch scores the same and the Viterbi pass picks by the order of its states
   - That order gave the whole stretch to the home team in both directions; the change is now placed in the middle of the stretch (23 change-overs moved, home-to-away 1.1 s earlier and away-to-home 0.5 s later on average)

4. **Restart logic:**
   - After each D stretch, check Smart Tagging Team tag for SET PIECES interval
   - If known, assign next ~2 seconds to that team
   - Handles kickoffs, goal kicks, corners (but not throw-ins tagged N/A)

4. **Pre-kick-off rule:**
   - Graded frames before a half's kick-off → `D`
   - For this match the second half is graded from frame 110,808 but kicks off at 113,742 (FAQ.md); those 2,934 frames are dead ball
   - Kick-off positions are the video anchors published in `games.py`

5. **X label assignment:**
   - Frames outside graded regions → `X`
   - Optionally: Low-confidence ball within possession → `X` (forward-fill in scorer)
   - We set `USE_X_FOR_OCCLUSIONS = False` (prefer H/A over X within possessions)

**Output:**
- `labels`: Array of strings {'H', 'A', 'D', 'X'} length 199,319

**Stats (full match):**
- H: 56,766 frames (28.5%)
- A: 36,464 frames (18.3%)
- D: 65,627 frames (32.9%; 41.3% of graded frames)
- X: 40,462 frames (20.3%)
- Total possessions: 228 (on full match)
- Mean possession duration: 16.4 s

---

### Stage 6: Weak Evaluation (`weak_eval.py`)

**Purpose:** Sanity check against Smart Tagging (not ground truth, but correlated).

**Metrics computed:**

1. **Team agreement:** On frames inside a Smart Tagging possession phase (the codes used in Stage 3) that we label live, what % carry the tagged team?
   - Result: 90.4% on full match
   - Partly circular, because those phases feed the labels. A gross-error check only.

2. **Dead agreement:** On SET PIECES intervals, what % of our labels are D?
   - Result: 30.8%. This is expected to be low: a SET PIECES tag is a 35 s window of which only the first few seconds are a stoppage. The number is logged but no longer means "agreement"

3. **Possession count and duration distribution:**
   - Used to catch gross errors (e.g., 10,000 possessions = clearly wrong)

**Output:** Diagnostics to `logs/diag.txt` only (not used in pipeline)

---

## Possession Rules

Our implementation follows these rules, derived from the challenge definition and public ground truth:

### Change of Possession

A possession change occurs when:
1. **Sustained control by opposing team:** an opposing player is on the ball (closest player, within about a stride) for roughly a second, and the previous team does not clearly have it back within the next second (hysteresis)
2. **After dead ball:** Possession assigned based on restarting team (Smart Tagging)
3. **While the ball is unseen:** only if a Smart Tagging possession phase for the other team persists for several seconds

### What is NOT a change of possession

- Single touch without control (deflection, rebound): one frame's evidence is capped
- Opponents pressing near the ball: only the closest player of each team is compared
- Ball in flight during pass (high speed, no control update)
- Loose ball (nobody within 2.5 m) - stays with last team
- Ball lost by the tracking - stays with last team

### Dead Ball

Ball is dead when:
- Out of bounds (outside pitch ± 2m)
- The players are collectively walking (strongly if the ball is not seen, mildly if it is)
- In the 3.5 s before a restart tagged by Smart Tagging (SET PIECES)
- Before the kick-off of a half

**After dead ball ends:** Next possession assigned to restart team (if known from Smart Tagging), else inferred spatially or from previous possession.

### Minimum Durations

- **Possession:** 1.5 seconds (realism constraint)
- **Dead ball:** 2 seconds (no real restart is quicker)

Shorter runs absorbed into neighbors during post-processing.

---

## Implementation Details

### Language and Libraries

- **Python 3.9 to 3.12** (`run.py` refuses newer versions; verified on 3.12)
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

- **Runtime:** two to three minutes (measured 132 to 181 s on a Windows 11 laptop)
- **Bottleneck:** ATD loading via kloppy

**Optimization opportunities (not pursued):**
- Parse ATD directly (skip kloppy DataFrame conversion)
- Vectorize evidence computation (currently frame-by-frame loop)
- Numba JIT for Viterbi inner loop

---

## Hyperparameters

All parameters in `src/possession/config.py`. **Chosen from football knowledge and the plausibility of the full-match output, not by optimising the public score** (17 possessions too small for reliable tuning). The public window's descriptive statistics (mean possession 14.2 s) were used as a reference for what a plausible output looks like.

### Ball Reconstruction

```python
BALL_SHORT_GAP_FRAMES = 25          # 1 second - typical touch/dribble duration
BALL_UNCERTAINTY_GROWTH_RATE = 0.1  # Exponential decay per frame
```

**Rationale:** 1-second gaps are typical occlusions (player blocking ball). Longer gaps suggest tracking loss or ball out of view.

### Control Evidence

```python
CONTROL_RADIUS_TAU = 1.0            # metres - a player controls a ball within about a stride
LOOSE_BALL_DISTANCE = 2.5           # metres - nobody this close: loose ball, no opinion
CONTROL_SCORE_CAP = 0.85            # a touch is not control: ~1 s of clear control to switch
BALL_CONFIDENCE_MIN = 0.3           # ball unseen for a while: no opinion
BALL_FLIGHT_SPEED_THRESHOLD = 8.0   # m/s (~29 km/h) - typical pass speed
POSSESSION_PHASE_CODES = [BUILD UP, PROGRESSION, ATTACKING TRANSITION, LONG BALL, BALL IN FINAL THIRD]
PHASE_PRIOR = 0.6                   # Smart Tagging opinion: 60/40
```

**Rationale:**
- 1 m radius, 2.5 m loose-ball distance: football judgement about what "on the ball" means, fixed before looking at any result
- 0.85 cap: derived from the transition cost so that a clean change of possession takes about a second (`40 / log(0.85/0.15) ≈ 23` frames)
- 0.6 phase prior: mild by design; at 0.6 a phase needs about 4 seconds of tracking silence to change the owner (`40 / log(0.6/0.4) ≈ 99` frames)
- 8 m/s: pass speed, unchanged from the first version
- None of these was swept against a score

### Dead Ball Detection

```python
COLLECTIVE_SPEED_DROP = 1.5         # m/s - players walking below, playing above
COLLECTIVE_SPEED_WIDTH = 0.1        # m/s - softness of that step
DEAD_PROB_WALKING_BALL_UNSEEN = 0.9 # players walking, ball lost: almost always a stoppage
DEAD_PROB_WALKING_BALL_SEEN = 0.6   # players walking, ball in view: more often than not
DEAD_PROB_ACTIVE_BALL_UNSEEN = 0.1  # players running: play is on
DEAD_PROB_ACTIVE_BALL_SEEN = 0.05
OUT_OF_BOUNDS_MARGIN = 2.0          # metres - generous margin for tracking noise
SET_PIECE_LEAD_IN_FRAMES = 87       # 3.5 s before a tagged restart
SET_PIECE_LEAD_IN_PROB = 0.95       # the ball is dead just before a restart
```

**Rationale:**
- The probabilities are rounded football judgements, not fitted values. Moving each of them (0.5-0.7 for walking/seen, 0.8-0.95 for walking/unseen, the step at 1.4-1.6 m/s) keeps the full match between 89 and 101 stoppages and 59 to 65 minutes of ball in play
- 2m margin accounts for tracking jitter near boundaries
- The 3.5 s lead-in is measured on the two kick-off tags against the kick-off anchors in `games.py`

### Viterbi Decoding

```python
TRANSITION_COST_H_A = 40.0          # High cost = sticky possessions
TRANSITION_COST_LIVE_TO_DEAD = 40.0 # Same as H<->A
TRANSITION_COST_DEAD_TO_LIVE = 40.0 # Same as H<->A
MIN_POSSESSION_FRAMES = 37          # ~1.5 seconds
MIN_DEAD_FRAMES = 50                # 2 seconds
HYSTERESIS_FRAMES = 25              # ~1.0 second sustained control required
```

**Rationale:**
- H↔A cost 40.0: Chosen by reviewing decoded sequences (prevents flicker while allowing real changes)
- Live↔dead cost 40.0: Set equal to the H↔A cost. Compared against 5, 10 and 20 on answer-key-free checks only; 40 is the only value giving a realistic number of stoppages (about 110; a match has roughly 100-130 restarts). Full table in UPDATES.md.
- Min durations: Possession informed by public GT statistics (mean possession 14.2s, shortest ~3s); stoppage from the fact that no restart takes under 2 s
- Hysteresis: Matches challenge definition of "sustained control"

**Parameter sensitivity:** not measured against a ground truth, because the public window is too small to do so reliably. The live↔dead cost is sensitive in terms of output shape (see the table in UPDATES.md).

---

## Limitations and Future Work

### Current Limitations

1. **Dead ball detection is approximate:**
   - D class F1 on the public window: 0.79 (precision 0.92, recall 0.69)
   - Stoppages start late: players keep running for two or three seconds after the ball goes out. Quick restarts where nobody walks are missed
   - The full match now has 62 minutes of ball in play, which is at the high end for a football match; some stoppages are probably still missed
   - When the ball is unseen, only player activity and Smart Tagging are left
   - A ball last seen out of bounds stays "out" until the tracking finds it again, which can run past the restart
   - On the public window we predict 17 possessions against 17 true ones, but find only 5 of 20 change-overs within a second: the right number of possessions with imprecise boundaries. Our misses are early as often as late, two true possessions last under 1.5 s, and in two cases the tracking shows the previous team still on the ball for seconds after the true change (see UPDATES.md, entry 19)

2. **We are blind for most of the match:**
   - Tracking has an opinion about who is on the ball in only 30% of frames; the ball is unseen in long gaps for 43%
   - In those stretches possession is held with the last team and can change only through a Smart Tagging phase or a stoppage, so real changes of possession during ball gaps are found late or missed
   - **What we would fix first:** estimate the ball's whereabouts during gaps from how the players move (both teams shift towards the ball), instead of staying silent

3. **No player identities:**
   - Cannot follow individual players through identity swaps
   - Limits ability to use player-specific models (e.g., "player X is a defender, less likely to score")
   - **Mitigation:** Use collective team features only

4. **Restart assignment imperfect:**
   - Smart Tagging Team = N/A for throw-ins (36 instances)
   - Fallback spatial heuristics (nearest touchline) unreliable
   - **Mitigation:** Short duration, limited impact on overall spellF1

5. **No video features:**
   - Referee signals (whistle, hand gestures) would improve dead ball detection
   - Player jersey numbers for identity
   - **Mitigation:** Tracking + Smart Tagging sufficient for first approach

6. **Hyperparameters are not validated against a ground truth:**
   - Transition costs, min-durations, and smoothing sigma interact non-linearly
   - Exhaustive grid search infeasible on 17-possession public GT
   - **Mitigation:** values chosen for football plausibility of the full-match output; their robustness has not been measured

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
   - Expected runtime: two to three minutes
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
   # Should show submission with spellF1 0.5176 and macroF1 0.8284 on the public window
   ```

### Determinism Guarantees

- **Fixed random seed:** `RANDOM_SEED = 42` in `config.py`
- **No randomness in pipeline:** All algorithms deterministic (Viterbi, interpolation, filters)
- **Pinned dependencies:** `requirements.txt` uses exact versions
- **No external API calls:** All computation local
- **No file ordering dependencies:** Loads single match only

**Verified:** two consecutive runs on Windows 11 (x64), Python 3.12, produce byte-identical files (SHA-256 `9f8f9abd80ac860a...`). Other platforms have not been tested; float rounding could in principle differ across architectures.

---

## Acknowledgments

- **Organizers:** Metrica Sports, FC Barcelona, Mobile World Capital Barcelona
- **Data:** LaLiga match (FC Barcelona vs Athletic Club, 2024-25 season, matchday 2)
- **Challenge design:** Framework for evaluating possession structure (spellF1 metric)

---

**Document version:** 1.2  
**Date:** November 12, 2026  
**Competition deadline:** November 17, 2026
