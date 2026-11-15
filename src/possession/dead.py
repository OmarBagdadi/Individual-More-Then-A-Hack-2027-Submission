"""Dead ball detection: per-frame probability that play has stopped.

Three signals, each with a football meaning:

1. What the players are doing. When play stops the players walk; while the ball is in play
   they run. The collective speed of the tracked players is the main signal.
2. Whether the ball is seen. Players walking while the tracking has lost the ball is a
   stoppage almost every time (the ball is in someone's hands or off the pitch). Players
   walking with the ball in view is weaker: it also happens in slow build-up play.
3. The rules. A ball seen beyond the lines is out. And a Smart Tagging SET PIECES tag marks a
   restart, so the ball was dead in the seconds just before it.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from src.possession.config import (
    COLLECTIVE_SPEED_DROP,
    COLLECTIVE_SPEED_WIDTH,
    COLLECTIVE_SPEED_SMOOTH_FRAMES,
    MIN_PLAYERS_FOR_SPEED,
    DEAD_PROB_WALKING_BALL_UNSEEN,
    DEAD_PROB_WALKING_BALL_SEEN,
    DEAD_PROB_ACTIVE_BALL_UNSEEN,
    DEAD_PROB_ACTIVE_BALL_SEEN,
    OUT_OF_BOUNDS_MARGIN,
    SET_PIECE_LEAD_IN_FRAMES,
    SET_PIECE_LEAD_IN_PROB,
    PITCH_LENGTH,
    PITCH_WIDTH,
    DEAD_BALL_CODES,
    LOG_FILE
)


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


""" Mean speed of the tracked players per frame, smoothed over about a second.

    Frames with too few tracked players (camera cuts, tracking drop-outs) take the last
    reliable value instead of an average over two or three players.

    Returns:
        array of speeds in m/s
"""
def collective_player_speed(tracking_df):
    speed_cols = [c for c in tracking_df.columns if c.endswith('_s')]
    player_speeds = tracking_df[speed_cols].values.astype(float)

    n_tracked = (~np.isnan(player_speeds)).sum(axis=1)
    with np.errstate(all='ignore'):
        mean_speed = np.nanmean(player_speeds, axis=1)
    mean_speed[n_tracked < MIN_PLAYERS_FOR_SPEED] = np.nan
    mean_speed = pd.Series(mean_speed).ffill().bfill().values

    return gaussian_filter1d(mean_speed, sigma=COLLECTIVE_SPEED_SMOOTH_FRAMES)


""" Compute per-frame probability that ball is dead (out of play).

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
    ball_seen = ball_state['visible']

    # Signal 1: are the players walking? 1 = clearly walking, 0 = clearly playing,
    # with a soft step around COLLECTIVE_SPEED_DROP.
    mean_speed = collective_player_speed(tracking_df)
    walking = 1.0 / (1.0 + np.exp((mean_speed - COLLECTIVE_SPEED_DROP) / COLLECTIVE_SPEED_WIDTH))

    # Signal 2: combine with whether the ball is seen.
    prob_if_walking = np.where(ball_seen, DEAD_PROB_WALKING_BALL_SEEN, DEAD_PROB_WALKING_BALL_UNSEEN)
    prob_if_active = np.where(ball_seen, DEAD_PROB_ACTIVE_BALL_SEEN, DEAD_PROB_ACTIVE_BALL_UNSEEN)
    dead_prob = walking * prob_if_walking + (1.0 - walking) * prob_if_active

    # Signal 3a: ball seen beyond the lines. Only where it is actually observed: in a tracking
    # gap the position is the last sighting, which says nothing about now.
    out_of_bounds = ball_seen & (
        (ball_x < -OUT_OF_BOUNDS_MARGIN) |
        (ball_x > PITCH_LENGTH + OUT_OF_BOUNDS_MARGIN) |
        (ball_y < -OUT_OF_BOUNDS_MARGIN) |
        (ball_y > PITCH_WIDTH + OUT_OF_BOUNDS_MARGIN)
    )
    dead_prob[out_of_bounds] = 1.0

    # Signal 3b: Smart Tagging SET PIECES. The tag is a fixed window that starts a few seconds
    # before the restart and runs on through the live play that follows (35 s for throw-ins,
    # corners, goal kicks and free kicks). Only its lead-in, before the restart, is dead ball.
    lead_in = np.zeros(n_frames, dtype=bool)
    for event in events:
        if event['code'] in DEAD_BALL_CODES:
            start = event['start_frame']
            end = min(start + SET_PIECE_LEAD_IN_FRAMES, event['end_frame'] + 1, n_frames)
            if start < n_frames:
                lead_in[start:end] = True
    dead_prob[lead_in] = np.maximum(dead_prob[lead_in], SET_PIECE_LEAD_IN_PROB)

    # Smooth probabilities to reduce flicker
    dead_prob = gaussian_filter1d(dead_prob, sigma=3.0)
    dead_prob = np.clip(dead_prob, 0, 1)

    # Statistics
    n_dead = (dead_prob > 0.5).sum()
    is_walking = walking > 0.5

    log(f"\nDead ball detection:")
    log(f"  Likely dead (prob > 0.5): {n_dead} frames ({100*n_dead/n_frames:.1f}%)")
    log(f"  Players walking, ball unseen: {(is_walking & ~ball_seen).sum()} frames")
    log(f"  Players walking, ball seen: {(is_walking & ball_seen).sum()} frames")
    log(f"  Ball seen out of bounds: {out_of_bounds.sum()} frames")
    log(f"  SET PIECES lead-in: {lead_in.sum()} frames")
    log(f"  Mean dead prob: {dead_prob.mean():.3f}")

    return dead_prob


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from src.possession.io import load_tracking, load_smart_tagging
    from src.possession.ball import reconstruct_ball

    print("Testing dead ball detection...")

    df, metadata, team_of_track = load_tracking()
    events = load_smart_tagging()

    ball_x = df['ball_x'].values
    ball_y = df['ball_y'].values
    ball_state = reconstruct_ball(ball_x, ball_y)

    dead_prob = compute_dead_ball_probability(df, ball_state, events)

    print(f"Dead frames (prob > 0.5): {(dead_prob > 0.5).sum()}")
    print(f"Alive frames (prob < 0.5): {(dead_prob < 0.5).sum()}")
    print(f"Mean dead probability: {dead_prob.mean():.3f}")
