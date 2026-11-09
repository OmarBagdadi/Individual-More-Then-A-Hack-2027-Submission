# Project Summary - More than a Hack 2027

**Status:** ✅ Complete and working  
**Date:** November 5, 2026  
**Deadline:** November 17, 2026 (12 days remaining)

---

## Quick Facts

- **Challenge:** Predict possession labels (H/A/D/X) for 199,319 video frames
- **Result:** spellF1 **0.4706** on public 5-minute window (beats all baselines)
- **Possessions:** 17 predicted (exactly matches ground truth count)
- **Runtime:** ~2 minutes
- **Approach:** Interpretable heuristics, no ML training

---

## What Was Built

### Core Pipeline (6 stages)
1. **io.py** - Data loading and alignment verification
2. **ball.py** - Ball trajectory reconstruction (handles 41% missing ball)
3. **evidence.py** - Distance-weighted control evidence from player positions
4. **dead.py** - Multi-signal dead ball detection
5. **decode.py** - Viterbi decoder with possession structure constraints
6. **weak_eval.py** - Smart Tagging comparison for sanity checks

### Documentation
- **README.md** - Quick start, results, architecture
- **TECHNICAL.md** - Full technical specification (15 pages)
- **requirements.txt** - Pinned dependencies for reproduction

### Output
- **evaluation/results/submission.csv** - Main deliverable (199,319 rows)
- **logs/diag.txt** - Diagnostic log with pipeline statistics

---

## Performance

### Public 5-Minute Window (17 possessions)

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| **spellF1** | **0.4706** | +44% over baseline_smooth50 |
| **Possessions** | 17 | Exact match (GT: 17) |
| **Mean duration** | 17.1s | Close (GT: 14.2s) |
| **Fragmentation** | 1.05 | Minimal (GT segments: 21, ours: 22) |
| **Team accuracy** | 90.4% | Strong |
| **Transition F1** | 0.2927 | Best of all submissions |

### Baseline Comparison

```
rank  submission                spellF1   poss     dur
----  ------------------------  -------  -----  ------
   1  OUR SUBMISSION             0.4706     17    17.1 ✓
   2  baseline_smooth125         0.4242     16    18.5
   3  baseline_smooth50          0.3256     26    10.8
   4  baseline_smooth25          0.2868     36     7.6
   5  baseline_raw               0.0378    131     2.0
```

---

## Key Design Decisions

1. **High H↔A transition cost** → Sticky possessions (hysteresis)
2. **Minimum duration constraints** → Prevents flicker
3. **Distance-weighted voting** → Robust to tracking errors
4. **Smart Tagging as prior only** → Not hard constraint
5. **Forward-fill long ball gaps** → Better than Kalman extrapolation

---

## What Works Well

✅ **Possession structure:** Matches ground truth count exactly  
✅ **Temporal coherence:** Low fragmentation (1.05)  
✅ **Team assignment:** 90% accuracy on alive frames  
✅ **Robustness:** Handles 41% missing ball gracefully  
✅ **Speed:** Runs in ~2 minutes  
✅ **Reproducibility:** Deterministic with pinned dependencies  

---

## Known Limitations

⚠️ **Dead ball detection weak** (F1 0.06)
- Over-predicts live play
- Intentional tradeoff for spellF1 optimization

⚠️ **No video features** 
- Could improve with referee signals, boundary detection

⚠️ **Smart Tagging Team = N/A for throw-ins**
- Falls back to spatial heuristics

---

## Files Changed

### New Files
```
run.py                          # Main entry point
src/possession/
  ├── __init__.py
  ├── config.py                 # All hyperparameters
  ├── io.py                     # Data loading
  ├── ball.py                   # Ball reconstruction
  ├── evidence.py               # Control evidence
  ├── dead.py                   # Dead ball detection
  ├── decode.py                 # Viterbi decoder
  ├── weak_eval.py              # Smart Tagging comparison
  └── submission.py             # CSV writer
TECHNICAL.md                    # Technical specification
logs/diag.txt                   # Diagnostic log
evaluation/results/submission.csv  # Output
```

### Modified Files
```
README.md                       # Complete rewrite
requirements.txt                # Pinned versions
.gitignore                      # Updated
```

---

## Testing Checklist

- [x] Pipeline runs end-to-end
- [x] Submission validates (format check passes)
- [x] Scores on public GT (ranks #1, spellF1 0.4706)
- [x] Possession count realistic (17 vs GT 17)
- [x] Duration distribution reasonable (mean 17.1s vs GT 14.2s)
- [x] Reproducible (deterministic with fixed seed)
- [x] Documentation complete (README + TECHNICAL)
- [x] Dependencies pinned (requirements.txt)

---

## Submission Ready

### Three Deliverables

1. ✅ **Possession file:** `evaluation/results/submission.csv`
   - 199,319 rows, format validated
   - Generates realistic possessions

2. ✅ **Code repository:** Current directory
   - Complete source in `src/possession/`
   - Executable: `python run.py`
   - Deterministically regenerates submission.csv

3. ✅ **Technical document:** `TECHNICAL.md`
   - 15 pages covering approach, assets, rules, limitations
   - Also: README.md for quick reference

### Before Submitting

- [ ] Push to public GitHub repository
- [ ] Verify data/ is excluded (.gitignore)
- [ ] Test on clean checkout
- [ ] Export TECHNICAL.md to PDF (optional, markdown accepted)
- [ ] Fill out Google Form with team name

---

## Next Steps (If Time Permits)

### Quick Wins (< 1 day)
1. **Tune transition costs** - Sweep 30-50 range for H↔A cost
2. **Adjust dead ball threshold** - Currently over-predicts live play
3. **Bootstrap confidence intervals** - Run `--bootstrap 1000` on public GT

### Medium Effort (2-3 days)
1. **Improve restart logic** - Better inference for throw-in N/A cases
2. **Add player formation features** - Detect defensive vs attacking shape
3. **Speed optimization** - Vectorize evidence loop, ~10x faster

### Research (1 week+)
1. **Broadcast video alignment** - Solve synchronization problem
2. **Video-based dead ball detection** - Referee pose classifier
3. **Learned model** - Train on synthetic data or augmented GT

---

## Production Readiness

**Current state:** Research prototype → Submission-ready  
**What it does well:** Core possession detection with structure  
**What it doesn't:** Video features, real-time inference, multi-match generalization  

**To productionize:**
- Add video-based features (referee, boundaries)
- Train model on multiple matches
- Optimize for real-time (< 1s per frame)
- Add confidence intervals per prediction
- Build API for live-match inference

---

## Contact

- **Repository:** (to be made public for submission)
- **Documentation:** README.md, TECHNICAL.md
- **Logs:** logs/diag.txt
- **Competition:** https://landing.mobileworldcapital.com/more-than-a-hack-2027

---

**Generated:** November 5, 2026  
**Pipeline version:** 1.0  
**Ready for submission:** Yes ✅
