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

# Distance decay parameter for control votes (metres)
# exp(-distance/tau) weighting for nearby players
CONTROL_DISTANCE_TAU = 3.0  # metres

# Radius for "nearby" players to vote (metres)
CONTROL_MAX_DISTANCE = 10.0  # beyond this, vote is negligible

# Threshold for "no clear control" signal
# If max_team_score / total_score < this, mark as contested
NO_CONTROL_THRESHOLD = 0.6  # require 60% dominance for clear control

# Speed threshold for "player is moving toward ball" bonus (m/s)
PLAYER_SPEED_THRESHOLD = 1.0  # m/s, ~3.6 km/h

# ============================================================================
# Dead ball detection (dead.py)
# ============================================================================

# Ball stationary threshold (m/s)
BALL_STATIONARY_SPEED = 0.3  # m/s

# Collective player speed drop threshold (m/s)
# Mean player speed below this suggests stoppage
COLLECTIVE_SPEED_DROP = 1.5  # m/s

# Out of bounds margins (metres beyond pitch boundaries)
OUT_OF_BOUNDS_MARGIN = 2.0  # metres

# Smart Tagging SET PIECES prior weight (0-1)
# How much to trust Smart Tagging for dead ball
SET_PIECES_PRIOR_WEIGHT = 0.7

# Restart signal: ball + players accelerating after stoppage
# Acceleration threshold (m/s²)
RESTART_ACCEL_THRESHOLD = 1.0  # m/s²

# ============================================================================
# Viterbi decoding (decode.py)
# ============================================================================

# Transition costs (log probabilities, negative = cost)
# H <-> A transition cost (high = hysteresis, require sustained control)
TRANSITION_COST_H_A = 40.0  # high cost = sticky possessions

# H/A <-> D transition costs
TRANSITION_COST_LIVE_TO_DEAD = 5.0
TRANSITION_COST_DEAD_TO_LIVE = 5.0

# Minimum possession duration (frames)
# Possessions shorter than this are absorbed into neighbors
MIN_POSSESSION_FRAMES = int(1.5 * FPS)  # ~1.5 seconds

# Minimum dead ball duration (frames)
MIN_DEAD_FRAMES = int(0.5 * FPS)  # ~0.5 seconds

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
