"""Evidence accumulation: compute per-frame soft control scores per team.

Distance-weighted voting from nearby players, gated by ball speed.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.possession.config import (
    CONTROL_DISTANCE_TAU,
    CONTROL_MAX_DISTANCE,
    NO_CONTROL_THRESHOLD,
    PLAYER_SPEED_THRESHOLD,
    BALL_FLIGHT_SPEED_THRESHOLD,
    LOG_FILE
)


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


""" Compute per-frame soft control scores for home and away teams.

    Strategy:
    - For each frame, players vote with weight exp(-distance/tau)
    - Gate updates when ball speed > threshold (passes in flight)
    - Output "no clear control" signal when dominance < threshold

    Args:
        tracking_df: DataFrame with tracking data
        ball_state: dict from ball.reconstruct_ball()
        metadata: dict with team info from io.load_tracking()
        team_of_track: array mapping track slot index to team (0=home, 1=away)

    Returns:
        dict with keys:
            home_score: array of home team control scores (0-1)
            away_score: array of away team control scores (0-1)
            no_control: boolean array, True when contested/unclear
"""
def compute_control_evidence(tracking_df, ball_state, metadata, team_of_track):
    n_frames = len(tracking_df)
    ball_x = ball_state['x']
    ball_y = ball_state['y']
    ball_speed = ball_state['speed']
    ball_confidence = ball_state['confidence']

    # Initialize scores
    home_scores = np.zeros(n_frames)
    away_scores = np.zeros(n_frames)
    no_control = np.zeros(n_frames, dtype=bool)

    # Get player position columns (numbered 1-46, not home_/away_ prefixed)
    # Format: 1_x, 1_y, 2_x, 2_y, ..., 46_x, 46_y
    n_slots = len(team_of_track)
    x_cols = [f'{i+1}_x' for i in range(n_slots)]
    y_cols = [f'{i+1}_y' for i in range(n_slots)]

    # Verify columns exist
    x_cols = [c for c in x_cols if c in tracking_df.columns]
    y_cols = [c for c in y_cols if c in tracking_df.columns]

    # Process frame by frame
    for i in range(n_frames):
        if np.isnan(ball_x[i]) or np.isnan(ball_y[i]):
            # No ball position, no control evidence
            no_control[i] = True
            continue

        # Ball in flight? Gate evidence accumulation
        if ball_speed[i] > BALL_FLIGHT_SPEED_THRESHOLD:
            # During fast ball movement, don't update control
            # Use previous frame's scores or neutral
            if i > 0:
                home_scores[i] = home_scores[i-1]
                away_scores[i] = away_scores[i-1]
            no_control[i] = True
            continue

        # Low confidence ball position? Mark as contested
        if ball_confidence[i] < 0.3:
            no_control[i] = True

        # Compute distance-weighted votes
        home_vote = 0.0
        away_vote = 0.0

        # All players (use team_of_track to assign votes)
        for slot_idx, (x_col, y_col) in enumerate(zip(x_cols, y_cols)):
            if slot_idx >= len(team_of_track):
                break

            px = tracking_df[x_col].iloc[i]
            py = tracking_df[y_col].iloc[i]

            if np.isnan(px) or np.isnan(py):
                continue

            dist = np.sqrt((px - ball_x[i])**2 + (py - ball_y[i])**2)

            if dist <= CONTROL_MAX_DISTANCE:
                weight = np.exp(-dist / CONTROL_DISTANCE_TAU)

                # Add to home or away vote based on team mapping
                if team_of_track[slot_idx] == 0:  # Home
                    home_vote += weight
                else:  # Away
                    away_vote += weight

        # Normalize votes to probabilities
        total_vote = home_vote + away_vote
        if total_vote > 0:
            home_scores[i] = home_vote / total_vote
            away_scores[i] = away_vote / total_vote

            # Check for clear control
            max_score = max(home_scores[i], away_scores[i])
            if max_score < NO_CONTROL_THRESHOLD:
                no_control[i] = True
        else:
            # No players nearby
            no_control[i] = True

    # Smooth scores slightly to reduce jitter
    home_scores = gaussian_filter1d(home_scores, sigma=2.0)
    away_scores = gaussian_filter1d(away_scores, sigma=2.0)

    # Re-normalize after smoothing
    for i in range(n_frames):
        total = home_scores[i] + away_scores[i]
        if total > 0:
            home_scores[i] /= total
            away_scores[i] /= total

    # Statistics
    n_clear = (~no_control).sum()
    n_home_dominant = (home_scores > away_scores).sum()
    n_away_dominant = (away_scores > home_scores).sum()

    log(f"\nControl evidence:")
    log(f"  Clear control: {n_clear} frames ({100*n_clear/n_frames:.1f}%)")
    log(f"  Home dominant: {n_home_dominant} frames ({100*n_home_dominant/n_frames:.1f}%)")
    log(f"  Away dominant: {n_away_dominant} frames ({100*n_away_dominant/n_frames:.1f}%)")
    log(f"  Mean home score: {home_scores.mean():.3f}")
    log(f"  Mean away score: {away_scores.mean():.3f}")

    return {
        'home_score': home_scores,
        'away_score': away_scores,
        'no_control': no_control,
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from src.possession.io import load_tracking
    from src.possession.ball import reconstruct_ball

    print("Testing evidence accumulation...")

    df, metadata = load_tracking()
    ball_x = df['ball_x'].values
    ball_y = df['ball_y'].values
    ball_state = reconstruct_ball(ball_x, ball_y)

    evidence = compute_control_evidence(df, ball_state, metadata)

    print(f"Home dominant: {(evidence['home_score'] > evidence['away_score']).sum()} frames")
    print(f"Away dominant: {(evidence['away_score'] > evidence['home_score']).sum()} frames")
    print(f"No clear control: {evidence['no_control'].sum()} frames")
