# Possession Labelling Pipeline - More than a Hack 2027

**Team Submission** for Phase 1: Ball Possession Challenge

A modular, interpretable system that predicts frame-by-frame possession labels (H/A/D/X) from automatic tracking data, Smart Tagging, and homography.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline (generates evaluation/results/submission.csv)
python run.py

# Validate the submission
python evaluation/validate_submission.py evaluation/results/submission.csv

# Score against public 5-minute ground truth
python evaluation/run_evaluation.py
```

**Expected runtime:** two to three minutes (measured: 132 to 181 s on a Windows 11 laptop)  
**Output:** `evaluation/results/submission.csv` (199,319 rows, ~1.9 MB)

A log of every change made since the first version is in [UPDATES.md](UPDATES.md).

---

## Results

Scored on the **public 5-minute window** (17 possessions, 7,500 graded frames):

| Metric | Our Submission | baseline_smooth50 (floor) |
|--------|----------------|---------------------------|
| **spellF1** | **0.5176** | 0.3256 (public) / 0.2433 (full match) |
| **macroF1** | **0.8284** | 0.6207 |
| **Possessions** (truth: 17) | 17 | 26 |
| **Mean duration** (truth: 14.2s) | 15.1s | 10.8s |
| **Home share of live time** (truth: 57.3%) | 52.4% | 62.1% |
| **Dead-ball F1** | 0.79 | 0.32 |
| **Team accuracy** | 89.0% | 85.8% |

**How to read this.** The public window is a correctness check, not a leaderboard: one
possession is worth about 6 points of spellF1 here, and the scorer's own bootstrap reports that
gaps below ~0.35 are within noise on 17 possessions. Our spellF1 is not distinguishable from
`baseline_smooth125` (0.4242) on this window; our macroF1 (5-95% range 0.69 to 0.89) is above it. The full-match score is unknown to us.

What we can check without an answer key, over the whole match: 228 possessions with a mean of
16.4 s (longest 94 s), 99 stoppages, the ball in play for 62 minutes, and the home team with
60.9% of live time in a match where it made two thirds of the passes. All are in the range of a
real football match.

---

## Reproduction

### Requirements

- **Python 3.9 to 3.12** (`run.py` refuses anything newer; verified on 3.12)
- **Data:** Enrolled participants receive the match data bundle. Unpack to `./data/` or set `MTAH_DATA_DIR`.

### Installation

```bash
pip install -r requirements.txt
```

Dependencies (pinned for reproducibility):
- `numpy==1.26.4`
- `pandas==2.2.2`
- `kloppy==3.18.0`
- `scipy==1.11.3`

### Execution

```bash
python run.py
```

This regenerates `evaluation/results/submission.csv` deterministically from the provided data.

**No configuration needed.** All hyperparameters are in `src/possession/config.py` with domain-knowledge priors (not tuned on public GT).

### Validation

```bash
# Check format compliance
python evaluation/validate_submission.py evaluation/results/submission.csv

# Score against public ground truth
python evaluation/run_evaluation.py

# Score against custom ground truth
python evaluation/run_evaluation.py --gt path/to/ground_truth.csv
```

---

## Architecture

The pipeline is a 6-stage modular system:

```
1. io.py           → Load ATD tracking, Smart Tagging, homography
2. ball.py         → Reconstruct ball trajectory (interpolate gaps, model uncertainty)
3. evidence.py     → Compute per-frame control scores (who is on the ball + Smart Tagging prior)
4. dead.py         → Detect dead ball periods (multi-signal fusion)
5. decode.py       → Viterbi decoding with hysteresis and min-duration constraints
6. weak_eval.py    → Compare with Smart Tagging (sanity check)
```

Each module logs diagnostics to `logs/diag.txt`.

### Key Design Decisions

1. **Ball reconstruction:** Short gaps (≤1s) interpolated with cubic splines. Long gaps forward-filled with exponentially decaying confidence.

2. **Control evidence:** The closest player of each team is compared: whoever is on the ball (within about a stride) has it, however many opponents press nearby. When nobody is on the ball, or the ball has not been seen for a while, tracking gives no opinion and possession stays with the last team. Smart Tagging possession phases add a mild 60/40 nudge. Gated during high ball speed (passes in flight).

3. **Dead ball detection:** What the players are doing is the main signal: when play stops they walk. Walking with the ball lost by the tracking is a near-certain stoppage; walking with the ball in view is a mild one. Plus two rules of the game: a ball seen beyond the lines is out, and the 3.5 s before a restart tagged by Smart Tagging are dead (the tag itself is a 35 s window that is mostly live play).

4. **Viterbi decoder:** 3-state (H/A/D) where every change of state carries the same high cost (hysteresis). Min-duration constraints prevent unrealistic flicker. A change of team that falls in a stretch with no evidence is placed in the middle of it.

5. **Restart logic:** Uses Smart Tagging Team tags to assign possession after dead ball periods.

6. **Pre-kick-off:** Graded frames before a half's kick-off are labelled dead (the second half is graded for 2,934 frames before it kicks off).

---

## Project Structure

```
.
├── run.py                          # Main pipeline entry point
├── src/possession/
│   ├── config.py                   # Hyperparameters (domain priors)
│   ├── io.py                       # Data loading and alignment verification
│   ├── ball.py                     # Ball trajectory reconstruction
│   ├── evidence.py                 # Control score computation
│   ├── dead.py                     # Dead ball detection
│   ├── decode.py                   # Viterbi sequence decoding
│   ├── weak_eval.py                # Smart Tagging comparison
│   └── submission.py               # CSV writer
├── evaluation/
│   ├── run_evaluation.py           # Official scorer
│   ├── validate_submission.py      # Format validator
│   └── results/
│       └── submission.csv          # Output (generated by run.py)
├── logs/
│   └── diag.txt                    # Diagnostic log (generated by run.py)
├── requirements.txt                # Pinned dependencies
├── README.md                       # This file
└── TECHNICAL.md                    # Detailed technical document
```

---

## Configuration

All hyperparameters are in `src/possession/config.py`:

```python
# Ball reconstruction
BALL_SHORT_GAP_FRAMES = 25          # 1 second at 25 fps

# Control evidence
CONTROL_RADIUS_TAU = 1.0            # metres, a player controls a ball within about a stride
LOOSE_BALL_DISTANCE = 2.5           # metres, nobody this close = loose ball, no opinion
CONTROL_SCORE_CAP = 0.85            # a touch is not control (~1 s of clear control to switch)
PHASE_PRIOR = 0.6                   # Smart Tagging possession phases: mild 60/40 opinion

# Dead ball
COLLECTIVE_SPEED_DROP = 1.5         # m/s, players walking below this, playing above
DEAD_PROB_WALKING_BALL_UNSEEN = 0.9 # players walking and the ball lost: stoppage
DEAD_PROB_WALKING_BALL_SEEN = 0.6   # players walking, ball in view: mild
SET_PIECE_LEAD_IN_FRAMES = 87       # 3.5 s before a tagged restart are dead

# Viterbi decoding
TRANSITION_COST_H_A = 40.0          # High cost = sticky possessions
TRANSITION_COST_LIVE_TO_DEAD = 40.0 # Same cost for starting / ending a stoppage
MIN_POSSESSION_FRAMES = 37          # ~1.5 seconds minimum
MIN_DEAD_FRAMES = 50                # 2 seconds minimum stoppage
HYSTERESIS_FRAMES = 25              # ~1.0 second sustained control required
```

These were chosen from football knowledge and from the plausibility of the full-match output (number of possessions, number and length of stoppages), **not by optimising the score on the public ground truth** (17 possessions is too small for reliable tuning). See [UPDATES.md](UPDATES.md) for the settings that were compared.

---

## Diagnostics

The pipeline writes diagnostics to `logs/diag.txt`:

- Data loading statistics
- Ball reconstruction: visible/reconstructed frames, gap counts
- Control evidence: dominant team distribution
- Dead ball detection: signal contributions
- Viterbi decoding: possession count, duration distribution
- Weak evaluation: Smart Tagging agreement

Example:

```
ATD tracking loaded: 199319 frames
  Ball visible: 113488 frames (56.9%)
  
Ball reconstruction:
  Short gaps (<=25 frames): 43 gaps, 185 frames
  Long gaps (>25 frames): 143 gaps, 85646 frames
  
Viterbi decoding:
  H: 56766 frames (28.5%)
  A: 36464 frames (18.3%)
  D: 65627 frames (32.9%)
  X: 40462 frames (20.3%)
  Total possessions: 228
  Mean possession duration: 16.4 s
```

---

## Limitations

1. **Dead ball detection is approximate:** On the public window it finds 69% of dead-ball frames with 92% precision (F1 0.79). Players keep running for two or three seconds after the ball goes out, so stoppages are found a little late, and a quick restart where nobody walks is missed. The full match comes out at 62 minutes of ball in play, at the high end for football.

2. **Hand-over timing is imprecise:** On the public window we predict 17 possessions against 17 true but find only 5 of 20 change-overs within a second; the misses are early as often as late, and some true possessions are shorter than the 1.5 s minimum. The tracking often does not show the moment the ball changes hands: nobody is clearly on the ball on about half of live frames.

3. **Tracking errors propagate:** Identity swaps and missing players in the ATD directly affect control evidence. Only team membership is used, never player identity.

4. **No video features:** The pipeline uses only tracking + Smart Tagging. Video analysis could improve dead ball detection and restart assignment.

5. **Smart Tagging dependency:** Restart logic and the possession prior rely on Smart Tagging Team tags, which are inferred and have N/A values for throw-ins. After a throw-in no restarting team is assigned; the decoder decides from the evidence.

6. **Hyperparameters are not validated against a ground truth:** there is none large enough. They are football judgements checked on the plausibility of the full-match output. The dead-ball probabilities were checked for stability: moving each of them keeps the match between 89 and 101 stoppages.

---

## What We Tried and Dropped

- **Kalman filtering for long gaps:** Too sensitive to ball acceleration noise. Simple forward-fill with confidence decay worked better.

- **Player speed bonuses:** Added weight for players moving toward the ball. Negligible impact on spellF1, removed for simplicity.

- **Broadcast video features:** Not synchronized to tactical timeline, complex to align. Left for future work.

- **ML models:** Interpretable heuristics reached competitive performance without training data.

---

## License and Citation

Code and documentation: MIT License (see LICENSE)

**Match data is not covered by this license.** It is separately licensed for competition use only under the terms accepted at enrolment.

```bibtex
@misc{mtah2027_submission,
  title  = {Possession Labelling Pipeline - More than a Hack 2027},
  year   = {2027},
  note   = {Phase 1: Ball Possession Challenge}
}
```

---

## Contact

For questions about the code or approach, open an issue in the repository.

For enrolment and data terms: **hackathon.talentarena@mobileworldcapital.com**
