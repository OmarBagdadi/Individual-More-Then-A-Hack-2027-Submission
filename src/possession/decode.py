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

    # Emission log-probabilities (higher = more likely; the forward pass maximises).
    emission = np.zeros((n_frames, n_states))

    # Live-play term shared by H and A: log P(ball is alive).
    log_alive = np.log(1 - dead_prob + 1e-6)

    # Home / away: the ball must be alive, plus the control evidence measured relative to
    # "no opinion" (score 0.5 -> 0). Who has the ball and whether it is in play are separate
    # questions: a frame with no control evidence must not look more dead than one with.
    emission[:, H] = np.log(2 * home_score + 1e-6) + log_alive
    emission[:, A] = np.log(2 * away_score + 1e-6) + log_alive

    # Dead state: log P(ball is dead).
    emission[:, D] = np.log(dead_prob + 1e-6)

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

    # A change of team decided inside a stretch with no evidence goes in the middle of it.
    path = center_changeovers(path, evidence['track_silent'], evidence['prior_sign'])

    # Use Smart Tagging to inform restart possessions
    path = apply_restart_logic(path, events)

    # The ball is dead from the start of each graded half until that half's kick-off.
    for half_start, kickoff in pre_kickoff_blocks():
        path[half_start:kickoff] = D

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


""" Frames between the start of each graded half and its kick-off.

    The graded region of a half can begin before the ball is kicked off (for this match the
    second half is graded from frame 110,808 but kicks off at 113,742, see FAQ.md). Play has
    not started, so those frames are dead ball. The kick-off positions are the video anchors
    published in games.py (minute, second, frame-within-second of the tactical video).

    Returns:
        list of (half_start_frame, kickoff_frame); kickoff_frame is exclusive
"""
def pre_kickoff_blocks():
    from games import GAMES, GAME, half_video_blocks

    game = GAMES[GAME]
    anchors = {1: game.get('kickoff'), 2: game.get('kickoff_2nd')}

    blocks = []
    for half, start_frame, end_frame in half_video_blocks(GAME):
        anchor = anchors.get(half)
        if anchor is None:
            continue
        # +1: the anchor is the last frame before the ball is in play
        kickoff = (anchor['minute'] * 60 + anchor['second']) * FPS + anchor['frame'] + 1
        if start_frame < kickoff <= end_frame:
            blocks.append((start_frame, kickoff))

    return blocks


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
            if not (away_score[window] >= home_score[window]).all():
                # Not sustained, revert to H
                path[i] = H
        elif path[i-1] == A and path[i] == H:
            # Switching from A to H
            window = slice(i, min(i + hysteresis_frames, n))
            if not (home_score[window] >= away_score[window]).all():
                # Not sustained, revert to A
                path[i] = A

    return path


""" Place a change of team in the middle of the silent stretch it falls in.

    When one team is on the ball, then nobody is (ball loose, in the air or unseen), then the
    other team is, the evidence does not say when inside that stretch the ball changed hands:
    every frame of it scores the same. The Viterbi pass then picks by the order of its states,
    which put home-to-away changes at the end of the stretch and away-to-home changes at the
    start, giving the whole stretch to the home team either way. The middle treats both teams
    alike and halves the worst-case timing error.

    A stretch only counts while the Smart Tagging opinion is constant across it; where Smart
    Tagging changes its mind inside the stretch the decoder has already used that.

    Args:
        path: array of state indices
        track_silent: boolean array, True where tracking has no opinion
        prior_sign: Smart Tagging phase opinion per frame

    Returns:
        modified path
"""
def center_changeovers(path, track_silent, prior_sign):
    path = path.copy()
    n = len(path)
    H, A, D = 0, 1, 2

    live_change = (path[1:] != path[:-1]) & (path[1:] != D) & (path[:-1] != D)
    shifts = {(H, A): [], (A, H): []}

    for i in np.flatnonzero(live_change) + 1:
        old, new = path[i - 1], path[i]
        if not (track_silent[i] or track_silent[i - 1]):
            continue
        if prior_sign[i - 1] != prior_sign[i]:
            continue

        start = i
        while (start > 0 and track_silent[start - 1] and path[start - 1] == old
               and prior_sign[start - 1] == prior_sign[i - 1]):
            start -= 1
        end = i
        while (end < n and track_silent[end] and path[end] == new
               and prior_sign[end] == prior_sign[i]):
            end += 1

        middle = (start + end) // 2
        path[start:middle] = old
        path[middle:end] = new
        shifts[(old, new)].append(middle - i)

    log("\nChange-overs centred in silent stretches:")
    for (old, new), name in (((H, A), 'home to away'), ((A, H), 'away to home')):
        moved = np.array(shifts[(old, new)])
        if len(moved):
            log(f"  {name}: {len(moved)} moved, mean shift {moved.mean() / FPS:+.2f} s")

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
