"""Hyperparameters for the possession labelling pipeline.

All parameters are chosen from priors and domain knowledge, never tuned on the public GT.
Prefer wide plateaus over narrow peaks.
"""

import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

# Pitch dimensions in metres (from games.py, measured from corrected tracking)
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

# Video properties
FPS = 25  # frames per second
N_FRAMES = 199319  # total tactical video frames

# ============================================================================
# Ball reconstruction (ball.py)
# ============================================================================

# Short gap interpolation threshold (frames). Gaps <= this are interpolated.
# At 25 fps: 25 frames = 1 second
BALL_SHORT_GAP_FRAMES = 25  # 1 second

# Long gap handling: carry last known position with growing uncertainty
BALL_UNCERTAINTY_GROWTH_RATE = 0.1  # per frame, dimensionless

# Kalman filter parameters for ball motion (used for interpolation)
BALL_PROCESS_NOISE = 0.5  # process noise covariance
BALL_MEASUREMENT_NOISE = 0.1  # measurement noise covariance

# Ball speed threshold for "ball in flight" (m/s)
# Above this, evidence accumulation is gated (passes/long balls)
BALL_FLIGHT_SPEED_THRESHOLD = 8.0  # ~29 km/h, typical pass speed

# ============================================================================
# Evidence accumulation (evidence.py)
# ============================================================================

# Control radius (metres). Only the closest player of each team counts, weighted
# exp(-distance/tau). A player controls a ball within about a stride; opponents pressing
# from a few metres away do not have it, however many they are.
CONTROL_RADIUS_TAU = 1.0  # metres

# If neither team has a player this close, the ball is loose or in flight and tracking has
# no opinion: possession stays with the last team (challenge definition).
LOOSE_BALL_DISTANCE = 2.5  # metres

# Strongest opinion a single frame may hold. A touch is not control: with this cap, about one
# second of clear control is needed to outweigh the cost of changing team
# (TRANSITION_COST_H_A / log(0.85/0.15) = 23 frames).
CONTROL_SCORE_CAP = 0.85

# Below this ball-position confidence the ball has not been seen for a while (long tracking
# gap, position frozen at the last sighting). Players standing near that stale spot say
# nothing about possession, so tracking has no opinion.
BALL_CONFIDENCE_MIN = 0.3

# Below this (absolute, log-odds) the smoothed tracking evidence counts as "no opinion".
# It only has to separate exact silence from the tails the smoothing leaves next to a touch.
SILENT_LOGIT = 0.05

# Smart Tagging phases whose code describes a team in possession. The tagged team is taken
# as a mild opinion (PHASE_PRIOR vs 1 - PHASE_PRIOR) that only decides when tracking is
# silent or balanced. Each of these has a mirror "DEFENDING ..." / "DEFENSIVE TRANSITION"
# phase for the other team over the same interval, which would only count the same thing
# twice. Box and chance phases are left out: both teams have the ball in them about equally.
POSSESSION_PHASE_CODES = [
    "BUILD UP",
    "PROGRESSION",
    "ATTACKING TRANSITION",
    "LONG BALL",
    "BALL IN FINAL THIRD",
]
PHASE_PRIOR = 0.6

# ============================================================================
# Dead ball detection (dead.py)
# ============================================================================

# What the players are doing is the main signal: when play stops they walk, while the ball is
# in play they run. The mean speed of the tracked players, smoothed over about a second,
# crosses from "walking" to "playing" around this speed (soft step of the given width).
COLLECTIVE_SPEED_DROP = 1.5  # m/s
COLLECTIVE_SPEED_WIDTH = 0.1  # m/s
COLLECTIVE_SPEED_SMOOTH_FRAMES = 25  # 1 second

# With fewer tracked players than this the mean speed is not trusted (camera cut, drop-out);
# the last reliable value is kept.
MIN_PLAYERS_FOR_SPEED = 8

# Probability that the ball is dead, by what the players do and whether the ball is seen.
# Football judgements, rounded, not fitted values:
# - players walking and the tracking has lost the ball: a stoppage almost every time
# - players walking with the ball in view: more often a stoppage than not, but slow build-up
#   play looks the same, so it takes several seconds of it to call a stoppage
# - players running: play is on, a little less surely when the ball is not seen
DEAD_PROB_WALKING_BALL_UNSEEN = 0.9
DEAD_PROB_WALKING_BALL_SEEN = 0.6
DEAD_PROB_ACTIVE_BALL_UNSEEN = 0.1
DEAD_PROB_ACTIVE_BALL_SEEN = 0.05

# Out of bounds margins (metres beyond pitch boundaries)
OUT_OF_BOUNDS_MARGIN = 2.0  # metres

# A Smart Tagging SET PIECES tag is a fixed window around a restart, not the stoppage itself:
# it starts a few seconds before the ball is put back in play and runs on through the live
# play that follows. Only this lead-in is treated as dead ball, and as near certain: the ball
# is dead just before a restart by the laws of the game. The length comes from the two
# kick-off tags, which start 82 and 90 frames before the kick-off anchors in games.py.
SET_PIECE_LEAD_IN_FRAMES = int(3.5 * FPS)
SET_PIECE_LEAD_IN_PROB = 0.95

# ============================================================================
# Viterbi decoding (decode.py)
# ============================================================================

# Transition costs (log probabilities, negative = cost)
# H <-> A transition cost (high = hysteresis, require sustained control)
TRANSITION_COST_H_A = 40.0  # high cost = sticky possessions

# H/A <-> D transition costs.
# Same as the H <-> A cost: every change of state is equally "expensive", so a stoppage needs
# sustained evidence just like a change of team does. A lower value (5.0 was used originally)
# produced about twice as many stoppages as a football match has.
TRANSITION_COST_LIVE_TO_DEAD = 40.0
TRANSITION_COST_DEAD_TO_LIVE = 40.0

# Minimum possession duration (frames)
# Possessions shorter than this are absorbed into neighbors
MIN_POSSESSION_FRAMES = int(1.5 * FPS)  # ~1.5 seconds

# Minimum dead ball duration (frames).
# No real stoppage (throw-in, free kick, goal kick, corner) is over in under 2 seconds.
MIN_DEAD_FRAMES = int(2.0 * FPS)  # 2 seconds

# Hysteresis duration for possession changes (frames)
# Require this many frames of sustained opposing control for H<->A switch
HYSTERESIS_FRAMES = int(1.0 * FPS)  # ~1.0 second

# ============================================================================
# Label assignment (decode.py)
# ============================================================================

# Graded play boundaries (video frames)
# From DATA.md and games.py
HALF_1_START = 11185
HALF_1_END = 90651
HALF_2_START = 110808
HALF_2_END = 190197

# Use X label outside graded regions
USE_X_OUTSIDE_GRADED = True

# Use X for in-possession ball occlusions (long gaps with high uncertainty)
USE_X_FOR_OCCLUSIONS = False  # Conservative: prefer forward-fill via X

# Ball confidence threshold for triggering X (if USE_X_FOR_OCCLUSIONS=True)
BALL_CONFIDENCE_THRESHOLD_FOR_X = 0.2

# ============================================================================
# Smart Tagging integration
# ============================================================================

# Team name mapping (Smart Tagging -> H/A)
SMART_TAGGING_HOME_TEAM = "Barcelona"
SMART_TAGGING_AWAY_TEAM = "Atletic Club"  # Note: typo in data, not "Athletic"

# Codes to consider for dead ball detection
DEAD_BALL_CODES = ["SET PIECES"]

# ============================================================================
# Logging and diagnostics
# ============================================================================

# Log file path
LOG_FILE = "logs/diag.txt"

# Log file encoding (use utf-8 to handle all characters)
LOG_ENCODING = "utf-8"

# Verbosity level (0=minimal, 1=normal, 2=verbose)
LOG_LEVEL = 1
