"""Evidence accumulation: compute per-frame soft control scores per team.

Three sources, in order of strength:

1. Who is on the ball. The closest player of each team is compared; a player within about a
   stride of the ball controls it, however many opponents are pressing nearby.
2. Nothing. If nobody is near the ball (loose, in flight) or the ball has not been seen for a
   while, tracking has no opinion and the scores are neutral (0.5 / 0.5). The decoder then keeps
   possession with the last team, which is what the challenge definition asks for.
3. Smart Tagging phases, as a mild nudge that only decides when tracking is silent or balanced.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from src.possession.config import (
    CONTROL_RADIUS_TAU,
    LOOSE_BALL_DISTANCE,
    CONTROL_SCORE_CAP,
    BALL_CONFIDENCE_MIN,
    SILENT_LOGIT,
    BALL_FLIGHT_SPEED_THRESHOLD,
    PHASE_PRIOR,
    POSSESSION_PHASE_CODES,
    SMART_TAGGING_HOME_TEAM,
    SMART_TAGGING_AWAY_TEAM,
    LOG_FILE
)


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def _logit(p):
    return np.log(p / (1 - p))


""" Distance from the ball to the closest player of each team, per frame.

    Args:
        tracking_df: DataFrame with tracking data (columns 1_x, 1_y, ... per track slot)
        ball_x, ball_y: ball position per frame (metres)
        team_of_track: array mapping track slot index to team (0=home, 1=away)

    Returns:
        (d_home, d_away): arrays in metres; np.inf where a team has no tracked player
"""
def nearest_player_distances(tracking_df, ball_x, ball_y, team_of_track):
    slots = [i for i in range(len(team_of_track))
             if f'{i+1}_x' in tracking_df.columns and f'{i+1}_y' in tracking_df.columns]
    px = tracking_df[[f'{i+1}_x' for i in slots]].values.astype(float)
    py = tracking_df[[f'{i+1}_y' for i in slots]].values.astype(float)
    teams = np.asarray(team_of_track)[slots]

    dist = np.sqrt((px - ball_x[:, None]) ** 2 + (py - ball_y[:, None]) ** 2)
    dist = np.where(np.isnan(dist), np.inf, dist)

    d_home = dist[:, teams == 0].min(axis=1)
    d_away = dist[:, teams == 1].min(axis=1)
    return d_home, d_away


""" Per-frame opinion of Smart Tagging about who has the ball.

    Only phases whose code describes a team in possession are used (POSSESSION_PHASE_CODES).
    Overlapping phases that disagree cancel out.

    Returns:
        array of +1 (home), -1 (away) or 0 (no opinion)
"""
def phase_prior_sign(events, n_frames):
    votes = np.zeros(n_frames)
    codes = {c.upper() for c in POSSESSION_PHASE_CODES}

    for event in events or []:
        if (event['code'] or '').strip().upper() not in codes:
            continue
        team = event['tags'].get('Team')
        if team == SMART_TAGGING_HOME_TEAM:
            sign = 1.0
        elif team == SMART_TAGGING_AWAY_TEAM:
            sign = -1.0
        else:
            continue
        start = max(event['start_frame'], 0)
        end = min(event['end_frame'] + 1, n_frames)
        votes[start:end] += sign

    return np.sign(votes)


""" Compute per-frame soft control scores for home and away teams.

    Args:
        tracking_df: DataFrame with tracking data
        ball_state: dict from ball.reconstruct_ball()
        metadata: dict with team info from io.load_tracking()
        team_of_track: array mapping track slot index to team (0=home, 1=away)
        events: Smart Tagging events from io.load_smart_tagging() (optional)

    Returns:
        dict with keys:
            home_score: array of home team control scores (0-1), 0.5 = no opinion
            away_score: array of away team control scores (0-1)
            no_control: boolean array, True where tracking has no opinion
            track_silent: boolean array, True where the smoothed tracking evidence is neutral
            prior_sign: Smart Tagging phase opinion per frame (+1 home, -1 away, 0 none)
"""
def compute_control_evidence(tracking_df, ball_state, metadata, team_of_track, events=None):
    n_frames = len(tracking_df)
    ball_speed = ball_state['speed']
    ball_confidence = ball_state['confidence']

    d_home, d_away = nearest_player_distances(
        tracking_df, ball_state['x'], ball_state['y'], team_of_track)

    # 1. Who is on the ball: closest player of each team, sharp fall-off with distance.
    #    logit(home) = (d_away - d_home) / tau is the same as normalising exp(-d / tau).
    with np.errstate(invalid='ignore'):
        track_logit = (d_away - d_home) / CONTROL_RADIUS_TAU
    track_logit = np.nan_to_num(track_logit, nan=0.0, posinf=np.inf, neginf=-np.inf)

    # A touch is not control: cap how strongly one frame can speak, so that about a second
    # of clear control is needed to outweigh the decoder's cost of changing team.
    cap = _logit(CONTROL_SCORE_CAP)
    track_logit = np.clip(track_logit, -cap, cap)

    # 2. No opinion when nobody is on the ball, or the ball position is not trustworthy.
    loose = np.minimum(d_home, d_away) > LOOSE_BALL_DISTANCE
    unseen = ball_confidence < BALL_CONFIDENCE_MIN
    missing = np.isnan(ball_state['x']) | np.isnan(ball_state['y'])
    no_control = loose | unseen | missing
    track_logit[no_control] = 0.0

    # Ball in flight (fast): keep the opinion held just before the pass or shot.
    in_flight = (ball_speed > BALL_FLIGHT_SPEED_THRESHOLD) & ~unseen & ~missing
    if in_flight.any():
        last_ok = np.where(~in_flight, np.arange(n_frames), 0)
        np.maximum.accumulate(last_ok, out=last_ok)
        track_logit = np.where(in_flight, track_logit[last_ok], track_logit)
        track_logit[in_flight & (last_ok == 0) & in_flight[0]] = 0.0

    # Smooth slightly to reduce jitter
    track_logit = gaussian_filter1d(track_logit, sigma=2.0)

    # 3. Smart Tagging phases: a mild nudge, decisive only where tracking is silent or balanced.
    prior_sign = phase_prior_sign(events, n_frames)
    total_logit = track_logit + _logit(PHASE_PRIOR) * prior_sign

    home_scores = 1.0 / (1.0 + np.exp(-total_logit))
    away_scores = 1.0 - home_scores

    # Statistics
    n_opinion = (~no_control).sum()
    n_home_dominant = (track_logit > 0).sum()
    n_away_dominant = (track_logit < 0).sum()

    log(f"\nControl evidence:")
    log(f"  Tracking has an opinion: {n_opinion} frames ({100*n_opinion/n_frames:.1f}%)")
    log(f"    Ball loose / nobody within {LOOSE_BALL_DISTANCE} m: {(loose & ~unseen).sum()} frames")
    log(f"    Ball not seen recently: {unseen.sum()} frames")
    log(f"  Home on the ball: {n_home_dominant} frames ({100*n_home_dominant/n_frames:.1f}%)")
    log(f"  Away on the ball: {n_away_dominant} frames ({100*n_away_dominant/n_frames:.1f}%)")
    log(f"  Smart Tagging phase prior: home {(prior_sign > 0).sum()} frames, "
        f"away {(prior_sign < 0).sum()} frames")

    return {
        'home_score': home_scores,
        'away_score': away_scores,
        'no_control': no_control,
        'track_silent': np.abs(track_logit) < SILENT_LOGIT,
        'prior_sign': prior_sign,
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from src.possession.io import load_tracking, load_smart_tagging
    from src.possession.ball import reconstruct_ball

    print("Testing evidence accumulation...")

    df, metadata, team_of_track = load_tracking()
    events = load_smart_tagging()
    ball_state = reconstruct_ball(df['ball_x'].values, df['ball_y'].values)

    evidence = compute_control_evidence(df, ball_state, metadata, team_of_track, events)

    print(f"Home dominant: {(evidence['home_score'] > evidence['away_score']).sum()} frames")
    print(f"Away dominant: {(evidence['away_score'] > evidence['home_score']).sum()} frames")
    print(f"No tracking opinion: {evidence['no_control'].sum()} frames")
