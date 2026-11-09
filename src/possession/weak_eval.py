"""Weak evaluation: compare our labels with Smart Tagging as a sanity check.

Not ground truth, but useful for catching gross errors.
"""

import numpy as np
from src.possession.config import (
    SMART_TAGGING_HOME_TEAM,
    SMART_TAGGING_AWAY_TEAM,
    DEAD_BALL_CODES,
    FPS,
    LOG_FILE
)


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def weak_eval(labels, events):
    """Compare predicted labels with Smart Tagging for sanity check.

    Metrics:
    - Agreement with Smart Tagging Team tags (on tagged intervals)
    - Agreement of D labels with SET PIECES intervals
    - Possession count and duration distribution
    - Dead ball share

    Args:
        labels: array of predicted labels
        events: Smart Tagging events

    Returns:
        dict with evaluation metrics
    """
    n_frames = len(labels)

    # Build Smart Tagging team map
    st_team = np.full(n_frames, '', dtype='U1')
    for event in events:
        team_tag = event['tags'].get('Team', 'N/A')
        if team_tag == SMART_TAGGING_HOME_TEAM:
            st_team[event['start_frame']:event['end_frame']+1] = 'H'
        elif team_tag == SMART_TAGGING_AWAY_TEAM:
            st_team[event['start_frame']:event['end_frame']+1] = 'A'

    # Build Smart Tagging SET PIECES map
    st_dead = np.zeros(n_frames, dtype=bool)
    for event in events:
        if event['code'] in DEAD_BALL_CODES:
            st_dead[event['start_frame']:event['end_frame']+1] = True

    # Compare team labels (where ST has a team tag)
    tagged_mask = (st_team == 'H') | (st_team == 'A')
    if tagged_mask.sum() > 0:
        team_agreement = (labels[tagged_mask] == st_team[tagged_mask]).sum() / tagged_mask.sum()
    else:
        team_agreement = np.nan

    # Compare dead ball labels
    pred_dead = (labels == 'D')
    if st_dead.sum() > 0:
        dead_agreement = (pred_dead[st_dead]).sum() / st_dead.sum()
    else:
        dead_agreement = np.nan

    # Possession statistics
    possessions = []
    current_team = None
    current_start = None

    for i, label in enumerate(labels):
        if label in ('H', 'A'):
            if label != current_team:
                # End previous possession
                if current_team is not None:
                    duration = (i - current_start) / FPS
                    possessions.append((current_team, current_start, i-1, duration))

                # Start new possession
                current_team = label
                current_start = i
        elif label == 'D':
            # End possession
            if current_team is not None:
                duration = (i - current_start) / FPS
                possessions.append((current_team, current_start, i-1, duration))
                current_team = None

    # Final possession
    if current_team is not None:
        duration = (n_frames - current_start) / FPS
        possessions.append((current_team, current_start, n_frames-1, duration))

    durations = [p[3] for p in possessions]
    home_poss = [p for p in possessions if p[0] == 'H']
    away_poss = [p for p in possessions if p[0] == 'A']

    # Dead ball share
    dead_share = pred_dead.sum() / n_frames

    log(f"\nWeak evaluation (Smart Tagging comparison):")
    log(f"  Team agreement (on tagged intervals): {team_agreement:.3f}")
    log(f"  Dead agreement (SET PIECES intervals): {dead_agreement:.3f}")
    log(f"  Total possessions: {len(possessions)}")
    log(f"    Home possessions: {len(home_poss)}")
    log(f"    Away possessions: {len(away_poss)}")
    if durations:
        log(f"  Duration: mean {np.mean(durations):.1f}s, "
            f"median {np.median(durations):.1f}s, "
            f"min {np.min(durations):.1f}s, max {np.max(durations):.1f}s")
    log(f"  Dead ball share: {dead_share:.3f}")

    return {
        'team_agreement': team_agreement,
        'dead_agreement': dead_agreement,
        'n_possessions': len(possessions),
        'n_home_possessions': len(home_poss),
        'n_away_possessions': len(away_poss),
        'durations': durations,
        'dead_share': dead_share,
    }


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from src.possession.io import load_tracking, load_smart_tagging, get_graded_mask
    from src.possession.ball import reconstruct_ball
    from src.possession.evidence import compute_control_evidence
    from src.possession.dead import compute_dead_ball_probability
    from src.possession.decode import viterbi_decode

    print("Testing weak evaluation...")

    df, metadata = load_tracking()
    events = load_smart_tagging()
    graded_mask = get_graded_mask()

    ball_x = df['ball_x'].values
    ball_y = df['ball_y'].values
    ball_state = reconstruct_ball(ball_x, ball_y)

    evidence = compute_control_evidence(df, ball_state, metadata)
    dead_prob = compute_dead_ball_probability(df, ball_state, events)
    labels = viterbi_decode(evidence, dead_prob, ball_state, events, graded_mask)

    metrics = weak_eval(labels, events)

    print(f"Team agreement: {metrics['team_agreement']:.3f}")
    print(f"Dead agreement: {metrics['dead_agreement']:.3f}")
    print(f"Possessions: {metrics['n_possessions']}")
