"""Dead ball detection: identify stoppages using multiple signals.

Combines ball out of bounds, stationary, collective player speed, Smart Tagging SET PIECES.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.possession.config import (
    BALL_STATIONARY_SPEED,
    COLLECTIVE_SPEED_DROP,
    OUT_OF_BOUNDS_MARGIN,
    SET_PIECES_PRIOR_WEIGHT,
    RESTART_ACCEL_THRESHOLD,
    PITCH_LENGTH,
    PITCH_WIDTH,
    FPS,
    DEAD_BALL_CODES,
    LOG_FILE
)


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


""" Compute per-frame probability that ball is dead (out of play).

    Signals:
    - Ball out of bounds (pitch boundaries + margin)
    - Ball stationary (speed < threshold)
    - Collective player speed drop (mean < threshold)
    - Smart Tagging SET PIECES intervals (prior)
    - Restart detection: ball + players accelerating

    Args:
        tracking_df: DataFrame with tracking data
        ball_state: dict from ball.reconstruct_ball()
        events: list of Smart Tagging events from io.load_smart_tagging()

    Returns:
        array of dead ball probabilities (0-1)
"""
def compute_dead_ball_probability(tracking_df, ball_state, events):
    n_frames = len(tracking_df)
    ball_x = ball_state['x']
    ball_y = ball_state['y']
    ball_speed = ball_state['speed']

    dead_prob = np.zeros(n_frames)

    # Signal 1: Ball out of bounds
    out_of_bounds = (
        (ball_x < -OUT_OF_BOUNDS_MARGIN) |
        (ball_x > PITCH_LENGTH + OUT_OF_BOUNDS_MARGIN) |
        (ball_y < -OUT_OF_BOUNDS_MARGIN) |
        (ball_y > PITCH_WIDTH + OUT_OF_BOUNDS_MARGIN)
    )
    dead_prob[out_of_bounds] = 1.0

    # Signal 2: Ball stationary
    stationary = ball_speed < BALL_STATIONARY_SPEED
    dead_prob[stationary] = np.maximum(dead_prob[stationary], 0.6)

    # Signal 3: Collective player speed drop
    # Get all player speed columns
    speed_cols = [c for c in tracking_df.columns if c.endswith('_s')]

    if speed_cols:
        # Compute mean player speed per frame
        player_speeds = tracking_df[speed_cols].values
        mean_speed = np.nanmean(player_speeds, axis=1)

        # Smooth to reduce noise
        mean_speed = gaussian_filter1d(mean_speed, sigma=5.0)

        low_activity = mean_speed < COLLECTIVE_SPEED_DROP
        dead_prob[low_activity] = np.maximum(dead_prob[low_activity], 0.4)

    # Signal 4: Smart Tagging SET PIECES prior
    set_pieces_mask = np.zeros(n_frames, dtype=bool)
    for event in events:
        if event['code'] in DEAD_BALL_CODES:
            start = event['start_frame']
            end = event['end_frame']
            if start < n_frames:
                end = min(end, n_frames - 1)
                set_pieces_mask[start:end+1] = True

    # Blend SET PIECES prior with other signals
    dead_prob[set_pieces_mask] = np.maximum(
        dead_prob[set_pieces_mask],
        SET_PIECES_PRIOR_WEIGHT
    )

    # Signal 5: Restart detection (inverse signal - ball dead UNTIL restart)
    # Look for sudden acceleration in ball + players after period of low activity
    if speed_cols:
        # Compute acceleration
        ball_accel = np.zeros(n_frames)
        ball_accel[1:] = np.diff(ball_speed) * FPS  # m/s²

        player_accel = np.zeros(n_frames)
        for i in range(1, n_frames):
            player_accel[i] = (mean_speed[i] - mean_speed[i-1]) * FPS

        # Restart = both ball and players accelerate after low activity
        restart = (
            (ball_accel > RESTART_ACCEL_THRESHOLD) &
            (player_accel > RESTART_ACCEL_THRESHOLD * 0.5)
        )

        # After restart, not dead for next ~3 seconds
        for i in np.where(restart)[0]:
            end_live = min(i + int(3 * FPS), n_frames)
            dead_prob[i:end_live] = np.minimum(dead_prob[i:end_live], 0.2)

    # Smooth probabilities to reduce flicker
    dead_prob = gaussian_filter1d(dead_prob, sigma=3.0)

    # Clip to [0, 1]
    dead_prob = np.clip(dead_prob, 0, 1)

    # Statistics
    n_dead = (dead_prob > 0.5).sum()
    n_set_pieces = set_pieces_mask.sum()

    log(f"\nDead ball detection:")
    log(f"  Likely dead (prob > 0.5): {n_dead} frames ({100*n_dead/n_frames:.1f}%)")
    log(f"  Out of bounds: {out_of_bounds.sum()} frames")
    log(f"  Ball stationary: {stationary.sum()} frames")
    log(f"  Low collective activity: {low_activity.sum()} frames")
    log(f"  SET PIECES intervals: {n_set_pieces} frames")
    log(f"  Mean dead prob: {dead_prob.mean():.3f}")

    return dead_prob


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from src.possession.io import load_tracking, load_smart_tagging
    from src.possession.ball import reconstruct_ball

    print("Testing dead ball detection...")

    df, metadata = load_tracking()
    events = load_smart_tagging()

    ball_x = df['ball_x'].values
    ball_y = df['ball_y'].values
    ball_state = reconstruct_ball(ball_x, ball_y)

    dead_prob = compute_dead_ball_probability(df, ball_state, events)

    print(f"Dead frames (prob > 0.5): {(dead_prob > 0.5).sum()}")
    print(f"Alive frames (prob < 0.5): {(dead_prob < 0.5).sum()}")
    print(f"Mean dead probability: {dead_prob.mean():.3f}")
