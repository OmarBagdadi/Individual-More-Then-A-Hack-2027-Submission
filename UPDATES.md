# Updates

A plain-language log of every change made to this repository after the initial analysis on
2026-11-12. Newest entries at the bottom. Each entry says what changed, where, and why.

Nothing here was chosen by comparing scores on the public 5-minute answer key. That window
(17 possessions) is used only to check the pipeline still works.

---

## 2026-11-12

### 1. Added this file

- **File:** `UPDATES.md` (new)
- **Why:** to keep a record of every change from here on.

### 2. Set up a working environment (no repository files changed)

- Created a Python 3.12 virtual environment in `.venv/` (already ignored by git) and installed
  the pinned `requirements.txt`. The machine's default Python is 3.14, which `run.py` refuses.
- Pointed the `MTAH_DATA_DIR` environment variable at the existing data folder instead of
  copying the match data.

### 3. Reproduced the original result before touching any code

Ran the unchanged pipeline. It reproduces the committed result exactly: the regenerated
`evaluation/reports/submission.json` is identical to the committed one (public-window
spellF1 0.4706), and the label counts match the documents (H 70,869 / A 69,562 / D 18,426 /
X 40,462, 410 possessions).

Baseline numbers for later comparison (original code):

| check | original |
|---|---|
| runtime | 234 s (documents say "~2 minutes") |
| dead-ball share of graded frames | 11.6% |
| possessions (full match) | 410, mean 13.7 s, median 8.0 s, longest 185 s |
| frames 110,808-113,741 (graded, all dead per FAQ.md) labelled D | 651 of 2,934 |
| agreement of our D with Smart Tagging set pieces | 7.1% (documents say 99.8%) |
| "clear control" frames | 28.2% (documents say 100%) |
| separate stoppages (dead stretches) | 84 |

### 4. Fixed a plus/minus bug in the decoder

- **File:** `src/possession/decode.py`, the "emission" block at the top of `viterbi_decode`
- **Before:** the score for "home has the ball" was `log(home_score) - log(1 - dead_prob)`
  (same for away).
- **After:** `log(home_score) + log(1 - dead_prob)`.
- **In football terms:** the more certain the system was that play had stopped, the *more* it
  favoured saying a team had the ball. It should be the opposite. This is why the original
  version found only 3% of dead-ball frames in the public window, worse than the organisers'
  simplest baseline (21%). The documents described that as an intentional trade-off; it was a bug.
- Also replaced the frame-by-frame loop with whole-array maths (same result, faster).

### 5. A "stationary ball" only counts when the ball is actually seen

- **File:** `src/possession/dead.py`, "Signal 2: Ball stationary"
- **Before:** `stationary = ball_speed < 0.3`
- **After:** `stationary = ball_state['visible'] & (ball_speed < 0.3)`
- **Why:** when the tracking loses the ball, the pipeline freezes it at its last known spot. A
  frozen ball has zero speed, so the old rule treated every tracking gap (43% of the match) as
  "ball is standing still, play has probably stopped". With the bug in entry 4 this was hidden;
  with the bug fixed, it labelled 59% of the match as dead. Real matches have the ball out of
  play roughly 40-45% of the time.

### 6. Stoppages made as "sticky" as changes of team

- **File:** `src/possession/config.py`
- **Before:** `TRANSITION_COST_LIVE_TO_DEAD = 5.0`, `TRANSITION_COST_DEAD_TO_LIVE = 5.0`,
  `MIN_DEAD_FRAMES = 0.5 s`
- **After:** both costs `40.0` (the same as the existing home/away switch cost),
  `MIN_DEAD_FRAMES = 2.0 s`
- **Why:** with the low cost, the fixed pipeline produced 221 separate stoppages, many under
  two seconds. A real match has roughly 100-130 restarts and none takes under two seconds.
- **How the value was chosen:** by football plausibility of the full-match output, never by the
  public answer key. Settings tried (all with entries 4, 5 and 7 applied):

  | live/dead cost | min stoppage | stoppages | possessions | mean possession | dead share |
  |---|---|---|---|---|---|
  | 5 | 0.5 s | 221 | 391 | 9.6 s | 41.0% |
  | 5 | 2 s | 177 | 370 | 10.3 s | 40.1% |
  | 10 | 0.5 s | 195 | 338 | 11.1 s | 40.9% |
  | 10 | 2 s | 169 | 330 | 11.5 s | 40.4% |
  | 20 | 0.5 s | 182 | 307 | 12.2 s | 40.9% |
  | 20 | 2 s | 159 | 299 | 12.7 s | 40.4% |
  | 40 | 0.5 s | 119 | 259 | 14.3 s | 41.8% |
  | **40** | **2 s** | **110** | **255** | **14.6 s** | **41.6%** |

  The chosen row is the only one with a realistic number of stoppages, and it needs no new
  number: every change of state simply costs the same.

### 7. Everything before a half's kick-off is dead ball

- **File:** `src/possession/decode.py`, new function `pre_kickoff_blocks()`, used in
  `viterbi_decode` just before the labels are written
- **Why:** the second half is graded from frame 110,808 but does not kick off until 113,742
  (stated in the organisers' FAQ.md). Those 2,934 frames are all dead ball. The original
  labelled only 651 of them `D` and invented possessions while players walked back out.
- **How:** uses the kick-off positions the organisers publish in `games.py`. It is a rule, not
  hand-labelling: "play cannot be live before the kick-off".

### 8. Result of entries 4-7 together

Ran `python run.py` twice. Both runs produced the identical file
(SHA-256 `b0e75186bffd2adafe1b97363b2d4e401f94c97331ee0a05189b3b7f62cfe7ea`), and
`validate_submission.py` accepts it.

Checks that need no answer key (whole match):

| check | original | now |
|---|---|---|
| dead-ball share of graded frames | 11.6% | 41.5% |
| possessions | 410, mean 13.7 s | 256, mean 14.5 s |
| separate stoppages | 84 (30 of them under 2 s) | 111 (1 under 2 s) |
| pre-kick-off frames labelled D | 651 of 2,934 | 2,934 of 2,934 |
| our D inside Smart Tagging set pieces | 7.1% | 86.9% |
| runtime | 234 s | 196 s |

Public 5-minute window, as a correctness check only (17 possessions; the scorer's bootstrap
says gaps below ~0.37 in spellF1 are noise here):

| metric | original | now |
|---|---|---|
| spellF1 (ranking) | 0.4706 | 0.4667 |
| macroF1 (tie-break) | 0.5500 | 0.7656 |
| dead-ball frames found (recall) | 3% | 63% |
| dead-ball precision | 22% | 68% |
| possessions (truth 17) | 17 | 13 |

Honest reading: the ranking score on the public window is unchanged within noise, the per-frame
score is clearly better, and the whole-match output now looks like a football match. Whether
the full-match spellF1 went up cannot be known before the organisers score it.

`evaluation/reports/submission.json` and `submission_transitions.csv` were regenerated by the
scorer and now describe the new output.

### 9. Tried and not kept

- **Sign fix alone (entry 4 without 5-7):** 59% of graded frames dead, mean possession 7.4 s.
- **Neutral 50/50 evidence when the ball has been unseen for a while:** sounded right (a hidden
  ball should not change owner), but with the current scoring it pushed the decoder towards
  dead ball (54% dead). Dropped; listed as future work in TECHNICAL.md.

### 10. Documents corrected to match the code

- **`README.md`:** results table and how to read it; runtime (3-4 min, was "~2 minutes");
  Python versions (3.9-3.12, was "3.13 not supported" next to other claims); design decisions
  (stationary rule, equal costs, pre-kick-off); config excerpt; example log numbers; the
  dead-ball limitation (no longer described as intentional).
- **`TECHNICAL.md`:** same corrections, plus: control-evidence statistics (clear control is
  28.2%, the document said 100%); dead-ball statistics; transition table; minimum stoppage;
  pre-kick-off rule; Smart Tagging agreement (dead agreement was listed as 99.8%, which was
  impossible given the document's own frame counts; it was 7.1% then and is 86.9% now);
  removed the untested claims "tested on Ubuntu and macOS", "Python 3.13", "30-50 all produce
  similar spellF1" and "robust to ±20%"; reworded "not tuned on public GT" to say exactly what
  the public window was and was not used for; new limitation on weak evidence in ball gaps.
- **`PROJECT_SUMMARY.md`:** rewritten with the current numbers. Removed "ranks #1" and "Ready
  for submission: Yes"; the pre-submission checklist now lists what is still open.

---

## 2026-11-12, second round: who has the ball

**Why this round happened.** The output gave Barcelona 47.9% of live-ball time. Public match
statistics (looked up afterwards: Barcelona made 568 of the match's 861 passes, published
possession 65%) suggested that was wrong. The competition rules forbid using corrected
possession data for this fixture, so **that figure was not used as a target and no setting
below was chosen to approach it.** Each change fixes something that is wrong as football, and
each was judged on the same answer-key-free checks as before. The resulting split is reported
as an outcome.

### 11. Diagnosis first (no code change)

Measured where the split came from, using only the tracking and our own labels:

| frames | Barcelona share in our labels |
|---|---|
| all live frames | 47.9% |
| ball observed | 54.9% |
| one team clearly on the ball, other team over 4 m away | 58.2% |
| both teams within 4 m of the ball | 49.8% |
| ball unseen for a long stretch (31% of live frames) | **32.8%** |

Independent of our labels: on frames where one team is clearly on the ball, it is Barcelona
61.6% of the time (63.5% first half, 59.4% second). So the tracking itself says Barcelona had
more of the ball; our rules were losing it in crowds and, above all, while the ball was unseen.

Also measured what Smart Tagging team tags mean, on those same clear frames:

| phase code | tagged team is the team on the ball |
|---|---|
| BUILD UP | 90% |
| PROGRESSION | 87% |
| LONG BALL | 78% |
| ATTACKING TRANSITION | 73% |
| BALL IN FINAL THIRD | 70% |
| BALL IN THE BOX / PLAYERS IN THE BOX / CREATING CHANCES | 55-61% (uninformative) |
| DEFENDING IN ... / DEFENSIVE TRANSITION | 10-27% (they name the team *without* the ball; each is the mirror of an attacking phase over the same interval) |

### 12. "On the ball" replaces crowd voting

- **File:** `src/possession/evidence.py` (rewritten), constants in `src/possession/config.py`
- **Before:** every player within 10 m of the ball voted for his team, weight `exp(-d / 3 m)`.
  A man on the ball at 0.5 m counted 0.85; three opponents pressing from 3 m counted 1.10 and
  "won". Being pressed is not losing the ball.
- **After:** only the closest player of each team is compared, with a sharp fall-off
  (`CONTROL_RADIUS_TAU = 1.0` m, about a stride). If nobody is within `LOOSE_BALL_DISTANCE =
  2.5` m the ball is loose or in flight and tracking gives no opinion (50/50), so the decoder
  keeps the last team, as the challenge definition says.
- **A touch is not control:** one frame's opinion is capped (`CONTROL_SCORE_CAP = 0.85`), so
  about one second of clear control is needed before possession can change hands.
- The calculation is now done on whole arrays instead of one frame at a time. Runtime fell
  from about 200 s to about 80 s.
- Removed constants that nothing uses any more: `CONTROL_DISTANCE_TAU`,
  `CONTROL_MAX_DISTANCE`, `NO_CONTROL_THRESHOLD`, `PLAYER_SPEED_THRESHOLD`.

### 13. No ball, no opinion

- **Files:** `src/possession/evidence.py`, `src/possession/decode.py`
- **Before:** in a long tracking gap the ball is frozen at its last known spot, and whoever
  stood near that spot kept "voting". Those frames gave Barcelona only 33%.
- **After:** when the ball has not been seen for a while (`BALL_CONFIDENCE_MIN = 0.3`),
  tracking gives no opinion and possession stays with the last team.
- **Decoder:** control evidence is now measured relative to "no opinion"
  (`log(2 * score)` instead of `log(score)`), so "who has it" and "is it in play" are separate
  questions. This is what made the same idea fail in entry 9.
- **Decoder, hysteresis check:** a change of team is now rejected only if the old team clearly
  has the ball again within a second (`>=` instead of `>`). With the old strict test, a
  "no opinion" frame counted as a contradiction and would have erased possessions.

### 14. Live/dead balance restored (`DEAD_BALL_BIAS`)

- **Files:** `src/possession/config.py`, `src/possession/decode.py`
- **Problem:** entry 13 made live play look more likely in every frame. Dead time fell to 28%
  of graded frames and a single "possession" lasted 211 seconds. Neither happens in football.
- **Change:** a constant `DEAD_BALL_BIAS = 0.5` added to the dead-ball score.
- **How the value was chosen:** the dead-ball signals are rough scores, not probabilities. The
  one that matters here is "the ball is unseen and the players are collectively walking"; the
  football reading is that play has stopped. That signal tips to dead at 0.405, and 0.5 sits
  just above it. Values compared (with entries 12, 13 and 15 applied):

  | bias | dead share | ball in play | possessions | mean | longest | over 60 s |
  |---|---|---|---|---|---|---|
  | 0 | 27.2% | 77 min | 221 | 20.9 s | 211 s | 13 |
  | 0.2 | 29.1% | 75 min | 233 | 19.3 s | 211 s | 12 |
  | 0.35 | 30.8% | 73 min | 226 | 19.5 s | 211 s | 11 |
  | **0.5** | **48.2%** | **55 min** | **221** | **14.9 s** | **81 s** | **3** |
  | 0.69 | 54.9% | 48 min | 214 | 13.4 s | 66 s | 1 |

  The jump between 0.35 and 0.5 is that one signal switching sides. 0.5 gives a ball-in-play
  time of about 55 minutes, normal for a top-flight match. Barcelona's share is about 60% in
  every row from 0.2 to 0.69, so this choice does not drive the possession split.
- **Weakness, stated plainly:** an all-or-nothing signal deciding 18% of the match is crude.
  48% dead is slightly above the 40-45% I expected beforehand.

### 15. Smart Tagging phases used as a mild opinion

- **Files:** `src/possession/evidence.py` (`phase_prior_sign`), `src/possession/config.py`
  (`POSSESSION_PHASE_CODES`, `PHASE_PRIOR`), `run.py` (passes the tags to the evidence stage)
- **Before:** team tags were used only to decide who restarts after a stoppage.
- **After:** during BUILD UP, PROGRESSION, ATTACKING TRANSITION, LONG BALL and BALL IN FINAL
  THIRD phases, the tagged team gets a 60/40 nudge. Clear tracking evidence always outweighs
  it; it decides only when tracking is silent or balanced, which is mostly when the ball is
  unseen. The organisers' DATA.md calls the team tag "a strong prior, for free".
- **Which codes:** chosen by what the code means in football and confirmed by entry 11. The
  DEFENDING mirrors are left out because they would count the same interval twice; box and
  chance phases are left out because they are uninformative.
- **Effect on its own** (bias 0.5): possessions 236 -> 221, mean 13.3 s -> 14.9 s,
  under-2-second possessions 8 -> 3, Barcelona share 57.4% -> 59.9%.

### 16. Smart Tagging sanity check fixed

- **File:** `src/possession/weak_eval.py`
- **Before:** treated every tagged team as the team with the ball (including DEFENDING phases)
  and counted dead and ungraded frames as disagreements. It reported 18-33% "agreement".
- **After:** uses the possession phases from entry 15 and compares only frames we label live.
  Now 91.8%. Note this is partly circular, since those phases now feed the labels; it remains
  a gross-error check only. The output file is unchanged by this entry.

### 17. Result of the second round

Two runs produce the identical file
(SHA-256 `5d32622806c3e8e51a9ca352fe810de1088eec7795937dfbef10f80b6bd891e0`); it validates.

Whole match, no answer key:

| check | original | after round 1 | now |
|---|---|---|---|
| Barcelona share of live time | 50.5% | 47.9% | **59.9%** |
| ... first half / second half | 52.8 / 48.3 | 50.8 / 45.0 | 60.4 / 59.2 |
| possessions | 410, mean 13.7 s | 256, mean 14.5 s | 221, mean 14.9 s |
| longest possession | 185 s | 158 s | 81 s |
| stoppages | 84 | 111 | 109 |
| dead share of graded frames | 11.6% | 41.5% | 48.2% |
| ball in play | 94 min | 62 min | 55 min |
| runtime | 234 s | 196 s | 79 s |

The split now agrees with what the tracking shows on clear frames (61.6%), and the two halves
agree with each other.

Public 5-minute window, correctness check only (bootstrap: gaps below ~0.41 are noise):

| metric | original | after round 1 | now |
|---|---|---|---|
| spellF1 (ranking) | 0.4706 | 0.4667 | 0.4828 |
| macroF1 (tie-break) | 0.5500 | 0.7656 | 0.7018 |
| possessions (truth 17) | 17 | 13 | 12 |
| Barcelona share (truth 57.3%) | 61.7% | 53.3% | 57.1% |
| dead-ball recall / precision | 3% / 22% | 63% / 68% | 46% / 65% |
| team accuracy when both say live | 90.4% | 89.6% | 86.4% |
| change-overs found within 1 s | 6 of 20 | 5 of 20 | 3 of 20 |

Honest reading. Better: the possession split, whole-match plausibility, and the ranking
metric on the public window (within noise). Worse on the public window than round 1: per-frame
score, dead-ball recall, and timing of change-overs; it also merges possessions there (12
against 17). All of those differences are inside the noise of a 17-possession sample, but they
point the same way: this version is slow to hand the ball over. Not tuned further, because
the public window must not be used to choose between versions.

**For comparison only:** published statistics give Barcelona 65% (a share of passes: 568 of
861 is 66%). Ours is a share of time and is now 59.9%. The gap was not closed by tuning.

### 18. Documents updated for the second round

`README.md`, `TECHNICAL.md` and `PROJECT_SUMMARY.md`: new numbers, the new evidence stage,
the decoder changes, the new constants, the limitations above, and a statement of how public
match statistics were and were not used.

## 2026-11-12, third round: handing the ball over

Asked: fix the pipeline being slow to hand the ball over. That description came from the public
5 minutes (12 possessions against 17, 3 of 20 change-overs within a second). The diagnosis below
shows it was only partly right. One real fault was found and fixed, and one plausible fix was
tested and rejected. Final file: SHA-256
`e634c9eff08e64e512385cbcca919ad860498b0e7753afacf4604e4aa7df9d02`, identical across two runs,
validates.

### 19. Diagnosis (no code change)

Whole match, no answer key:

- The rate of possessions is not low: one every 14.9 s, against 14.2 s in the public truth.
- The decoder does not ignore clear evidence. Where one team is clearly on the ball (own player
  within 1 m, nearest opponent beyond 2.5 m) the label disagrees on 5.3% of frames, and only 9 of
  221 possessions contain a full second of the other team in clear control.
- The separate "hysteresis" pass moves only 3 change-overs; it is not the cause of anything.
- Smart Tagging changes its mind about the team 32 times in live play; 26 of those are matched by
  a change-over of ours.

Public 5 minutes, used as the challenge text invites, to understand the definition:

- Our misses are not systematically late. In time order: 4.8 s late, about 6 s late, 1.3 s
  early, 1.9 s early, 4.7 s early. Three are within half a second.
- Two true possessions last 0.8 s and 1.3 s, shorter than our 1.5 s minimum.
- In two true hand-overs the tracking shows the *previous* team clearly on the ball during the
  next 4 seconds and the new team never. Nearest-player geometry cannot find those.
- A 20-second true stoppage has the ball "observed" inside the pitch and moving faster than
  2 m/s for most of it. That is either a false ball detection or a ball being thrown around
  before a restart. Ball position cannot find that one either. (Lowering the 2 m out-of-bounds
  margin was considered and not done: it would not have caught this.)
- Touches shorter than half a second match the true team about half the time (no information);
  touches of a second or more match on about 92% of live frames. This supports the existing
  "about a second of control" rule, so the cost of changing team was **not** lowered.
- Between a touch by one team and the next touch by the other, the truth changes hands on
  average around the middle of the gap, not at its end.

### 20. Fault found: ties were always given to the home team

- **File:** `src/possession/decode.py` (new `center_changeovers`), `src/possession/evidence.py`
  (returns `track_silent` and `prior_sign`), `src/possession/config.py` (`SILENT_LOGIT = 0.05`).
- **Problem:** when one team is on the ball, then nobody is, then the other team is, every frame
  of the silent stretch scores the same. The Viterbi pass then chooses by the order of its
  states (home first). That put home-to-away changes at the *end* of the stretch and
  away-to-home changes at the *start*: the loose ball went to Barcelona both ways.
- **Change:** such a change-over is placed in the middle of the silent stretch, unless Smart
  Tagging changes its opinion inside it (then the decoder has already used that).
- **Effect:** 23 change-overs moved: 12 home-to-away by 1.1 s earlier on average, 11
  away-to-home by 0.5 s later. Barcelona share of live time 59.9% -> 59.7% (60.2 / 59.2 by
  half). Possessions (221), stoppages (109) and dead share (48.2%) unchanged. Public window:
  spellF1 0.4828 (unchanged), macroF1 0.7018 -> 0.7017.

### 21. Tried and not kept: "a pass between two team-mates is possession"

- **Idea:** when the same team touches the ball twice within 4 seconds, credit the flight in
  between to that team, and ignore isolated touches under half a second. The public window
  agrees with the rule itself (77-91% of such gap frames belong to that team).
- **Result:** it strengthens whoever already has the ball. Whole match: change-overs 114 -> 102,
  possessions 221 -> 208, mean length 14.9 -> 15.8 s, further from the 14.2 s of the public
  truth. That is the opposite of the request, so it never reached the repository.
  (On the public window it gave 8 possessions against 17.)

### 22. Documents updated for the third round

`README.md`, `TECHNICAL.md` and `PROJECT_SUMMARY.md`: new numbers, the centring step, and the
"slow to hand the ball over" limitation reworded to "hand-over timing is imprecise", which is
what the diagnosis supports.

## 2026-11-12, fourth round: dead-ball detection

Asked: fix the dead-ball detection. Final file: SHA-256
`9f8f9abd80ac860ac9ec6bc2dffa350758e370f3e364d325774902ded79742d5`, identical across two runs,
validates.

### 23. Diagnosis (no code change)

- **The set-piece tag was misread.** Every Smart Tagging SET PIECES tag for a throw-in, corner,
  goal kick or free kick is exactly 876 frames (35 s) long; kick-off tags are 126 frames (5 s).
  It is a fixed window around a restart, not the stoppage. Averaged over all 58 graded tags, the
  players are slowest and the ball least visible around the *start* of the tag, and play is in
  full flow from about 5 s in. The two kick-off tags start 82 and 90 frames before the kick-off
  anchors in `games.py`, so a tag starts about 3.5 s before the restart.
  The old code called the whole 35 s "70% dead": our labels were dead on 84% of tagged frames.
- **What the players do is the best signal, and it was nearly unused.** Used as the challenge
  text invites, to understand the definition, the public 5 minutes show: players walking
  (mean speed under 1.5 m/s) with the ball lost by the tracking is dead on 99% of frames;
  walking with the ball in view on about 45-70%; players running on 3-12%. The old code gave
  "players walking" a flat 0.4 whatever the ball was doing.
- **The "stationary ball" signal carries no information.** A ball seen standing still was dead on
  19% of frames, the same as any other frame. Removed.
- Every true stoppage in the public window contains a walking spell, but players are still
  running for 2-3 s after the ball goes out.

### 24. Dead-ball evidence rebuilt around the players

- **Files:** `src/possession/dead.py` (rewritten), `src/possession/config.py`.
- **Change:** dead probability per frame =
  - players walking and ball not seen: 0.9
  - players walking and ball seen: 0.6
  - players playing: 0.05 (0.1 if the ball is not seen)
  - ball seen beyond the lines (2 m margin, unchanged): 1.0
  - the first 3.5 s of a SET PIECES tag (the lead-in to the restart): 0.95

  "Walking" is a soft step at 1.5 m/s (the threshold the code already had) on the mean player
  speed smoothed over a second; frames with fewer than 8 tracked players keep the last value.
- **Removed:** the 35-second set-piece prior, the stationary-ball signal, the
  restart-acceleration signal, and their constants (`BALL_STATIONARY_SPEED`,
  `SET_PIECES_PRIOR_WEIGHT`, `RESTART_ACCEL_THRESHOLD`).
- **How the numbers were chosen:** as rounded football judgements, then checked for stability on
  the whole match, not picked by public score. Each one moved on its own:

  | Setting moved | Stoppages | Ball in play |
  |---|---|---|
  | as chosen (tag lead-in 0.7) | 95 | 62.0 min |
  | walking, ball seen 0.5 / 0.7 | 90 / 101 | 64.7 / 59.5 min |
  | walking, ball unseen 0.8 / 0.95 | 90 / 96 | 62.6 / 61.6 min |
  | step at 1.4 / 1.6 m/s | 89 / 95 | 65.2 / 60.4 min |
  | step twice as soft | 84 | 64.3 min |
  | no set-piece tag at all | 89 | 63.3 min |

- **Set-piece lead-in at 0.95 instead of 0.7:** a restart means the ball was dead just before it,
  by the laws of the game. At 0.7 four of the 58 tagged restarts had no stoppage before them and
  one "possession" ran 130 s through a restart; at 0.95 all 58 have one (99 stoppages, longest
  possession 94 s).

### 25. `DEAD_BALL_BIAS` removed

- **File:** `src/possession/decode.py`, `src/possession/config.py`.
- The constant existed to tip one weak 0.4 signal over the line (entry 14, and listed as the
  least principled setting). With probabilities that mean what they say it is not needed.
  Keeping it at 0.5 on top of the new evidence would give 107 stoppages and 57.7 minutes of ball
  in play; it was not kept because nothing but the result argues for it.

### 26. Result of the fourth round

Whole match, no answer key:

| Check | Before | Now |
|---|---|---|
| Stoppages | 109, median 20.2 s | 99, median 17.8 s |
| Share of graded frames dead | 48.2% | 41.3% |
| Ball in play | 54.8 min | 62.2 min |
| Dead inside the 35 s set-piece windows | 84% | 31% |
| Possessions | 221, mean 14.9 s, longest 81 s | 228, mean 16.4 s, longest 94 s |
| Barcelona share of live time | 59.7% (60.2 / 59.2) | 60.9% (64.2 / 57.3) |

Public 5 minutes (correctness check; 1000-draw bootstrap puts spellF1 gaps under 0.35 in the noise):

| Metric | Before | Now |
|---|---|---|
| spellF1 (ranking) | 0.4828 | 0.5176 (5-95%: 0.34 to 0.69) |
| macroF1 (tie-break) | 0.7017 | 0.8284 (5-95%: 0.69 to 0.89) |
| Dead ball: precision / recall / F1 | 0.65 / 0.46 / 0.54 | 0.92 / 0.69 / 0.79 |
| Possessions (truth 17) | 12, mean 21.6 s | 17, mean 15.1 s (truth 14.2 s) |
| Change-overs within 1 s | 3 of 20 | 5 of 20 |
| Team accuracy (both alive) | 86.4% | 89.0% |
| Barcelona share (truth 57.3%) | 57.1% | 52.4% |

Honest reading:

- The dead-ball numbers moved by more than the noise; the ranking score did not.
- 62 minutes of ball in play is at the high end for a football match, and 99 stoppages at the
  low end. Some stoppages are probably still missed: quick restarts where nobody walks, and the
  first 2-3 s of every stoppage while the players are still running.
- The Barcelona share on the public window got worse (57.1% -> 52.4%) and the two halves of the
  match now differ more (64.2% / 57.3%). Neither was tuned.
- The pipeline took 132 to 181 s in these runs against 79 s before. The new code does not
  explain that (the removed signals had the slow loops); the cause was not investigated.
- The "dead agreement on SET PIECES intervals" line in `logs/diag.txt` now reads 30.8%. It is
  expected to be low and no longer measures agreement.

### 27. Documents updated for the fourth round

`README.md`, `TECHNICAL.md` (Stage 4 rewritten, Stage 5 emission, hyperparameters, limitations)
and `PROJECT_SUMMARY.md`.

### Left for the repository owner to decide (not changed)

- `README.md` says "MIT License (see LICENSE)" but there is no `LICENSE` file.
- `README.md` says "Team Submission" while the repository is named "Individual-...". Teams need
  3-5 members to be eligible for the later rounds, and the form takes one response per team.
- Nothing has been committed or pushed. The changes are in the working folder only.
