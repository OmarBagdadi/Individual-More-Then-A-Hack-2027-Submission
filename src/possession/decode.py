"""Viterbi decoding: 3-state (H/A/D) sequence labelling with transition costs.

High H<->A cost for hysteresis, min-duration constraints, restart-based transitions.
"""

import numpy as np
from src.possession.config import (
    TRANSITION_COST_H_A,
    TRANSITION_COST_LIVE_TO_DEAD,
    TRANSITION_COST_DEAD_TO_LIVE,
    MIN_POSSESSION_FRAMES,
    MIN_DEAD_FRAMES,
    HYSTERESIS_FRAMES,
    HALF_1_START,
    HALF_1_END,
    HALF_2_START,
    HALF_2_END,
    USE_X_OUTSIDE_GRADED,
    USE_X_FOR_OCCLUSIONS,
    BALL_CONFIDENCE_THRESHOLD_FOR_X,
    SMART_TAGGING_HOME_TEAM,
    SMART_TAGGING_AWAY_TEAM,
    FPS,
    LOG_FILE
)


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


"""Decode possession sequence using Viterbi algorithm with possession structure.

    States: 0=H, 1=A, 2=D
    Transition costs enforce hysteresis and min-duration constraints.

    Args:
        evidence: dict from evidence.compute_control_evidence()
        dead_prob: array from dead.compute_dead_ball_probability()
        ball_state: dict from ball.reconstruct_ball()
        events: list from io.load_smart_tagging()
        graded_mask: boolean array from io.get_graded_mask()

    Returns:
        array of labels ('H', 'A', 'D', 'X')
"""
def viterbi_decode(evidence, dead_prob, ball_state, events, graded_mask):
    n_frames = len(dead_prob)
    home_score = evidence['home_score']
    away_score = evidence['away_score']

    # States: 0=H, 1=A, 2=D
    H, A, D = 0, 1, 2
    n_states = 3

    # Emission log-probabilities (negative = cost)
    # Higher score = lower cost
    emission = np.zeros((n_frames, n_states))

    for i in range(n_frames):
        # Home state: cost inversely proportional to home score
        emission[i, H] = -(-np.log(home_score[i] + 1e-6) + np.log(1 - dead_prob[i] + 1e-6))

        # Away state: cost inversely proportional to away score
        emission[i, A] = -(-np.log(away_score[i] + 1e-6) + np.log(1 - dead_prob[i] + 1e-6))

        # Dead state: cost inversely proportional to dead probability
        emission[i, D] = -(-np.log(dead_prob[i] + 1e-6))

    # Transition costs
    transition = np.array([
        # From:  H                        A                        D
        [0.0,                     TRANSITION_COST_H_A,     TRANSITION_COST_LIVE_TO_DEAD],  # To H
        [TRANSITION_COST_H_A,     0.0,                     TRANSITION_COST_LIVE_TO_DEAD],  # To A
        [TRANSITION_COST_DEAD_TO_LIVE, TRANSITION_COST_DEAD_TO_LIVE, 0.0]                  # To D
    ])

    # Viterbi forward pass
    dp = np.full((n_frames, n_states), -np.inf)
    backpointer = np.zeros((n_frames, n_states), dtype=int)

    # Initialize first frame
    dp[0, :] = emission[0, :]

    # Forward pass
    for i in range(1, n_frames):
        for j in range(n_states):
            # For each current state j, find best previous state
            costs = dp[i-1, :] + emission[i, j] - transition[:, j]
            best_prev = np.argmax(costs)
            dp[i, j] = costs[best_prev]
            backpointer[i, j] = best_prev

    # Backward pass to find best path
    path = np.zeros(n_frames, dtype=int)
    path[-1] = np.argmax(dp[-1, :])

    for i in range(n_frames - 2, -1, -1):
        path[i] = backpointer[i+1, path[i+1]]

    # Post-processing: enforce min-duration constraints
    path = enforce_min_duration(path, MIN_POSSESSION_FRAMES, [H, A])
    path = enforce_min_duration(path, MIN_DEAD_FRAMES, [D])

    # Apply hysteresis for H<->A transitions
    path = apply_hysteresis(path, HYSTERESIS_FRAMES, home_score, away_score)

    # Use Smart Tagging to inform restart possessions
    path = apply_restart_logic(path, events)

    # Convert to labels
    label_map = {H: 'H', A: 'A', D: 'D'}
    labels = np.array([label_map[s] for s in path])

    # Apply X labels
    if USE_X_OUTSIDE_GRADED:
        labels[~graded_mask] = 'X'

    if USE_X_FOR_OCCLUSIONS:
        # Use X for in-possession ball occlusions with low confidence
        low_conf = ball_state['confidence'] < BALL_CONFIDENCE_THRESHOLD_FOR_X
        # Only X if within a possession (not at transitions)
        for i in range(1, n_frames - 1):
            if low_conf[i] and labels[i-1] == labels[i+1] and labels[i] != 'D':
                labels[i] = 'X'

    # Statistics
    n_h = (labels == 'H').sum()
    n_a = (labels == 'A').sum()
    n_d = (labels == 'D').sum()
    n_x = (labels == 'X').sum()

    # Count possessions
    possessions = count_possessions(labels)

    log(f"\nViterbi decoding:")
    log(f"  H: {n_h} frames ({100*n_h/n_frames:.1f}%)")
    log(f"  A: {n_a} frames ({100*n_a/n_frames:.1f}%)")
    log(f"  D: {n_d} frames ({100*n_d/n_frames:.1f}%)")
    log(f"  X: {n_x} frames ({100*n_x/n_frames:.1f}%)")
    log(f"  Total possessions: {possessions}")
    log(f"  Mean possession duration: {(n_h + n_a) / (possessions + 1e-6) / FPS:.1f} s")

    return labels


""" Absorb runs shorter than min_frames into neighbors.

    Args:
        path: array of state indices
        min_frames: minimum run length
        states: list of states to apply this constraint to

    Returns:
        modified path
"""
def enforce_min_duration(path, min_frames, states):
    path = path.copy()
    n = len(path)

    # Find runs
    runs = []
    start = 0
    for i in range(1, n + 1):
        if i == n or path[i] != path[start]:
            runs.append((start, i - 1, path[start]))
            if i < n:
                start = i

    # Absorb short runs
    for start, end, state in runs:
        if state not in states:
            continue

        length = end - start + 1
        if length < min_frames:
            # Absorb into neighbors
            # Simple strategy: use previous state if available, else next
            if start > 0:
                path[start:end+1] = path[start-1]
            elif end < n - 1:
                path[start:end+1] = path[end+1]

    return path


""" Apply hysteresis to H<->A transitions.

    Require sustained opposing control for hysteresis_frames before switching.

    Args:
        path: array of state indices
        hysteresis_frames: number of frames of sustained control required
        home_score: home control scores
        away_score: away control scores

    Returns:
        modified path
"""
def apply_hysteresis(path, hysteresis_frames, home_score, away_score):
    path = path.copy()
    n = len(path)
    H, A, D = 0, 1, 2

    # For each H<->A transition, check if it's sustained
    for i in range(hysteresis_frames, n - hysteresis_frames):
        if path[i-1] == H and path[i] == A:
            # Switching from H to A, check if A dominant for next hysteresis_frames
            window = slice(i, min(i + hysteresis_frames, n))
            if not (away_score[window] > home_score[window]).all():
                # Not sustained, revert to H
                path[i] = H
        elif path[i-1] == A and path[i] == H:
            # Switching from A to H
            window = slice(i, min(i + hysteresis_frames, n))
            if not (home_score[window] > away_score[window]).all():
                # Not sustained, revert to A
                path[i] = A

    return path


""" Use Smart Tagging to assign restart possessions after dead balls.

    After a D stretch, check if Smart Tagging Team tag tells us who restarted.

    Args:
        path: array of state indices
        events: Smart Tagging events

    Returns:
        modified path
"""
def apply_restart_logic(path, events):
    path = path.copy()
    n = len(path)
    H, A, D = 0, 1, 2

    # Build interval map: frame -> team (if SET PIECES)
    set_pieces_team = {}
    for event in events:
        if event['code'] == 'SET PIECES':
            team_tag = event['tags'].get('Team', 'N/A')
            if team_tag == SMART_TAGGING_HOME_TEAM:
                team = H
            elif team_tag == SMART_TAGGING_AWAY_TEAM:
                team = A
            else:
                team = None  # N/A or unclear

            if team is not None:
                for frame in range(event['start_frame'], event['end_frame'] + 1):
                    if frame < n:
                        set_pieces_team[frame] = team

    # Find D stretches and check what comes after
    i = 0
    while i < n:
        if path[i] == D:
            # Find end of D stretch
            j = i
            while j < n and path[j] == D:
                j += 1

            # What team restarts?
            if j < n:
                # Check Smart Tagging for this interval
                restart_team = None
                for k in range(i, j):
                    if k in set_pieces_team:
                        restart_team = set_pieces_team[k]
                        break

                # If we know who restarts, use it
                if restart_team is not None:
                    # Set next few frames to restarting team
                    end_restart = min(j + int(2 * FPS), n)
                    path[j:end_restart] = restart_team

            i = j
        else:
            i += 1

    return path


""" Count number of possessions (maximal H or A runs).

    Args:
        labels: array of label strings

    Returns:
        number of possessions
"""
def count_possessions(labels):
    count = 0
    prev = None

    for label in labels:
        if label in ('H', 'A'):
            if prev != label:
                count += 1
            prev = label
        elif label == 'D':
            prev = None  # D breaks possessions

    return count


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    from src.possession.io import load_tracking, load_smart_tagging, get_graded_mask
    from src.possession.ball import reconstruct_ball
    from src.possession.evidence import compute_control_evidence
    from src.possession.dead import compute_dead_ball_probability

    print("Testing Viterbi decoding...")

    df, metadata = load_tracking()
    events = load_smart_tagging()
    graded_mask = get_graded_mask()

    ball_x = df['ball_x'].values
    ball_y = df['ball_y'].values
    ball_state = reconstruct_ball(ball_x, ball_y)

    evidence = compute_control_evidence(df, ball_state, metadata)
    dead_prob = compute_dead_ball_probability(df, ball_state, events)

    labels = viterbi_decode(evidence, dead_prob, ball_state, events, graded_mask)

    print(f"Labels: H={np.sum(labels=='H')}, A={np.sum(labels=='A')}, "
          f"D={np.sum(labels=='D')}, X={np.sum(labels=='X')}")
    print(f"Possessions: {count_possessions(labels)}")
