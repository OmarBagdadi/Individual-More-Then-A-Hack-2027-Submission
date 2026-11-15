# Project Summary - More than a Hack 2027

**Status:** Pipeline runs end to end and the output validates. Not yet submitted.  
**Updated:** November 12, 2026 (first version: November 5, 2026)  
**Deadline:** November 17, 2026

Every change since the first version is listed in [UPDATES.md](UPDATES.md).

---

## Quick Facts

- **Challenge:** Predict possession labels (H/A/D/X) for 199,319 video frames
- **Public 5-minute window:** spellF1 **0.5176**, macroF1 **0.8284** (a correctness check, not a ranking)
- **Full match:** score unknown; 228 possessions predicted, mean 16.4 s, home team 60.9% of live time
- **Runtime:** two to three minutes
- **Approach:** Interpretable heuristics, no ML training

---

## What Was Built

### Core Pipeline (6 stages)
1. **io.py** - Data loading and alignment verification
2. **ball.py** - Ball trajectory reconstruction (handles 41% missing ball)
3. **evidence.py** - Who is on the ball (closest player per team), silence when unknown, Smart Tagging phases as a mild prior
4. **dead.py** - Dead ball detection: players walking, ball seen or not, out of bounds, the seconds before a tagged restart
5. **decode.py** - Viterbi decoder with possession structure constraints
6. **weak_eval.py** - Smart Tagging comparison for sanity checks

### Documentation
- **README.md** - Quick start, results, architecture
- **TECHNICAL.md** - Technical document (the third deliverable)
- **UPDATES.md** - Change log
- **requirements.txt** - Pinned dependencies for reproduction

### Output (generated, not committed)
- **evaluation/results/submission.csv** - Main deliverable (199,319 rows)
- **logs/diag.txt** - Diagnostic log with pipeline statistics

---

## Performance

### Public 5-Minute Window (17 possessions)

| Metric | Value | Ground truth / note |
|--------|-------|---------------------|
| **spellF1** | **0.5176** | ranking metric |
| **macroF1** | **0.8284** | tie-break; every baseline is at or below 0.62 |
| **Possessions** | 17 | 17 |
| **Mean duration** | 15.1s | 14.2s |
| **Home share of live time** | 52.4% | 57.3% |
| **Fragmentation** | 1.10 | 23 predicted segments vs 21 |
| **Team accuracy** | 89.0% | on frames where both say the ball is alive |
| **Dead-ball F1** | 0.79 | precision 0.92, recall 0.69 |
| **Transition F1** | 0.24 | 5 of 20 change-overs found within ±1 s |

### Public-window board

```
rank  submission                macroF1  spellF1   poss     dur
----  ------------------------  -------  -------  -----  ------
   1  submission                 0.8284   0.5176     17    15.1
   2  baseline_smooth125         0.5812   0.4242     16    18.5
   3  baseline_smooth50          0.6207   0.3256     26    10.8
   4  baseline_smooth25          0.6200   0.2868     36     7.6
   5  baseline_raw               0.5972   0.0378    131     2.0
```

**Read with care.** On 17 possessions the scorer's bootstrap puts spellF1 gaps below ~0.35 inside
the noise, so first place here proves nothing about the full match. The organisers' own example:
`baseline_smooth125` beats `baseline_smooth50` on this window and loses to it on the full match.

### Full-match plausibility (no answer key)

| Check | Value | What a real match looks like |
|-------|-------|------------------------------|
| Possessions | 228, mean 16.4 s, median 9.7 s, longest 94 s | public truth averages 14.2 s |
| Stoppages | 99, median 17.8 s | roughly 100-130 restarts |
| Ball in play | 62 min (41.3% of graded frames dead) | about 55 min in a top-flight match |
| Home share of live time | 60.9% (64.2% / 57.3% by half) | tracking on clear frames: 61.6% |
| Frames before 2nd-half kick-off labelled D | 2,934 of 2,934 | all dead (FAQ.md) |

---

## Key Design Decisions

1. **Who is on the ball, not who is near it** → the closest player of each team is compared
2. **No ball, no opinion** → possession stays with the last team when the ball is loose or unseen
3. **Every change of state costs the same (40)** → Sticky possessions and sticky stoppages
4. **Minimum duration constraints** → Prevents flicker
5. **Smart Tagging as mild prior only** → Not hard constraint
6. **Dead before kick-off** → Uses the kick-off anchors in `games.py`
7. **No external statistics used** → public match stats were looked at, never used as a target

---

## Known Limitations

- **Hand-over timing is imprecise.** On the public window we predict 17 possessions against 17
  but find only 5 of 20 change-overs within a second, early as often as late.
- **Dead ball detection is approximate** (F1 0.79 on the public window). Stoppages are found two
  or three seconds late and quick restarts are missed; 62 minutes of ball in play is at the high
  end for a football match.
- **No video features.**
- **Smart Tagging Team = N/A for throw-ins**, so the restarting team is not known there.
- **Hyperparameters are not validated against a ground truth**; there is none large enough.

---

## Testing Checklist

- [x] Pipeline runs end-to-end (Python 3.12, Windows 11)
- [x] Submission validates (format check passes)
- [x] Two consecutive runs produce byte-identical files
- [x] Scores on public GT as a correctness check
- [x] Full-match possession, stoppage and ball-in-play numbers plausible
- [x] Documentation matches the code (README + TECHNICAL)
- [x] Dependencies pinned (requirements.txt)
- [ ] Clean-checkout test on a second machine

---

## Before Submitting

- [ ] Agree the changes in UPDATES.md with the repository owner, commit and push
- [ ] Confirm `data/` and the submission CSV are not in the public repository
- [ ] Re-run `python run.py`, then rename the output to `<team_name>.csv`
- [ ] Run `python evaluation/validate_submission.py <team_name>.csv` on that exact file
- [ ] Save the technical document as `<team_name>.pdf` or `<team_name>.md`
- [ ] Add a `LICENSE` file or remove the licence line from README.md
- [ ] Fill out the Google Form (one response per team) with the registered team name

---

## Contact

- **Documentation:** README.md, TECHNICAL.md, UPDATES.md
- **Logs:** logs/diag.txt
- **Competition:** https://landing.mobileworldcapital.com/more-than-a-hack-2027
