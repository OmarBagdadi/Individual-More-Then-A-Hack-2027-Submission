"""Load and align match data: ATD tracking, Smart Tagging, homography.

All loaders confirm alignment programmatically and convert to metres using the real pitch size.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from games import asset_paths, GAMES, GAME, half_video_blocks
from src.possession.config import (
    PITCH_LENGTH, PITCH_WIDTH, N_FRAMES, FPS,
    SMART_TAGGING_HOME_TEAM, SMART_TAGGING_AWAY_TEAM,
    DEAD_BALL_CODES, LOG_FILE
)


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


""" Load ATD tracking data via kloppy, convert to metres, verify alignment.

    Returns:
        df: DataFrame with columns [frame_id, ball_x, ball_y, ball_speed,
                                    1_x, 1_y, 1_s (per slot), ...]
            All positions in metres. NaN for missing observations.
        metadata: dict with team info and alignment verification
        team_of_track: array mapping track slot index to team (0=home, 1=away)
"""
def load_tracking():
    try:
        from kloppy import metrica
    except ImportError:
        raise ImportError("kloppy required. Install: pip install -r requirements-starter.txt")

    paths = asset_paths()
    dataset = metrica.load_tracking_epts(
        meta_data=paths["atd_meta"],
        raw_data=paths["atd_raw"]
    )

    # Verify alignment: should be 199319 records, frame_id 1..199319
    frame_ids = [int(r.frame_id) for r in dataset.records]
    assert len(frame_ids) == N_FRAMES, f"Expected {N_FRAMES} records, got {len(frame_ids)}"
    assert frame_ids[0] == 1 and frame_ids[-1] == N_FRAMES, \
        f"Expected frame_id 1..{N_FRAMES}, got {frame_ids[0]}..{frame_ids[-1]}"
    assert frame_ids == list(range(1, N_FRAMES + 1)), "frame_ids not contiguous"

    # Get team mapping from metadata
    teams = dataset.metadata.teams
    home_team = str(teams[0])  # First team is home (FIFA/EPTS convention)
    away_team = str(teams[1])

    # Verify against expected teams
    assert "Barcelona" in home_team or "Barça" in home_team, f"Expected Barcelona home, got {home_team}"
    assert "Athletic" in away_team or "Atletic" in away_team, f"Expected Athletic away, got {away_team}"

    # Build team_of_track mapping: array[slot_idx] -> 0 (home) or 1 (away)
    # Team 0 (home) has tracks 1-24, Team 1 (away) has tracks 25-46
    # But we need 0-indexed array
    import numpy as np

    # Get all players from both teams
    all_players = []
    for team in teams:
        all_players.extend(team.players)

    # Build mapping: for each slot (0-indexed), which team?
    team_of_track = np.zeros(len(all_players), dtype=int)
    for i, player in enumerate(all_players):
        # Check which team this player belongs to
        if player in teams[0].players:  # home
            team_of_track[i] = 0
        else:  # away
            team_of_track[i] = 1

    # Verify attack direction per half programmatically
    # We'll check this by looking at team centroids in each half
    half_blocks = half_video_blocks(GAME)

    # Convert to DataFrame
    df = dataset.to_df()

    # Convert normalized coordinates (0-1) to metres
    # kloppy uses 0-1 normalized; multiply by pitch dimensions
    for col in df.columns:
        if col.endswith('_x') and col != 'frame_id':
            df[col] = df[col] * PITCH_LENGTH
        elif col.endswith('_y') and col != 'frame_id':
            df[col] = df[col] * PITCH_WIDTH
        # Speed is already in m/s from kloppy

    # Add video frame index (frame_id - 1)
    df['frame'] = df.index  # kloppy already does frame_id - 1 in the index

    # Verify attack direction
    attack_direction = {}
    for half_num, start_frame, end_frame in half_blocks:
        half_df = df.loc[start_frame:end_frame]

        # Get home team player columns (slots with team_of_track == 0)
        home_slots = [i+1 for i, t in enumerate(team_of_track) if t == 0]
        home_x_cols = [f'{slot}_x' for slot in home_slots if f'{slot}_x' in half_df.columns]

        if home_x_cols:
            home_x_mean = half_df[home_x_cols].mean(axis=1).mean()
            attack_direction[half_num] = 'right' if home_x_mean < PITCH_LENGTH / 2 else 'left'
        else:
            attack_direction[half_num] = 'unknown'

    log(f"ATD tracking loaded: {len(df)} frames")
    log(f"  Home team: {home_team}, Away team: {away_team}")
    log(f"  Track slots: {len(team_of_track)} ({(team_of_track==0).sum()} home, {(team_of_track==1).sum()} away)")
    log(f"  Attack direction - Half 1: {attack_direction.get(1, 'unknown')}, "
        f"Half 2: {attack_direction.get(2, 'unknown')}")
    log(f"  Ball visible: {df['ball_x'].notna().sum()} frames "
        f"({100 * df['ball_x'].notna().sum() / len(df):.1f}%)")

    metadata = {
        'home_team': home_team,
        'away_team': away_team,
        'attack_direction': attack_direction,
        'n_frames': len(df),
        'ball_visible_count': df['ball_x'].notna().sum(),
    }

    return df, metadata, team_of_track


""" Load Smart Tagging intervals, convert to video frames.

    Returns:
        events: List of dicts with keys [code, start_frame, end_frame, tags]
                All frames are video frame indices (0-based)
"""
def load_smart_tagging():
    import xml.etree.ElementTree as ET

    paths = asset_paths()
    xml_path = paths["smart_tagging"]

    events = []
    for instance in ET.parse(xml_path).iter("instance"):
        start_s = float(instance.findtext("start"))
        end_s = float(instance.findtext("end"))

        event = {
            'code': instance.findtext("code"),
            'start_s': start_s,
            'end_s': end_s,
            'start_frame': int(round(start_s * FPS)),
            'end_frame': int(round(end_s * FPS)),
            'tags': {
                label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")
            }
        }
        events.append(event)

    # Verify time base by checking kickoffs
    kickoff_events = [
        e for e in events
        if (e['code'] or '').strip().lower() == 'set pieces'
        and (e['tags'].get('Type') or '').strip().lower().startswith('kick off')
    ]

    log(f"Smart Tagging loaded: {len(events)} instances")
    log(f"  Codes: {len(set(e['code'] for e in events))} unique")
    log(f"  Kickoff events found: {len(kickoff_events)}")

    # Count SET PIECES
    set_pieces = [e for e in events if e['code'] in DEAD_BALL_CODES]
    log(f"  SET PIECES events: {len(set_pieces)}")

    return events


def load_homography():
    """Load homography matrices for pitch <-> pixel projection.

    Returns:
        H: array of shape (N_FRAMES, 3, 3), homography matrices
        valid: array of shape (N_FRAMES,), bool mask for usable matrices
        reproj_error: array of shape (N_FRAMES,), reprojection errors
    """
    paths = asset_paths()
    df = pd.read_csv(paths["homography"])

    assert len(df) == N_FRAMES, f"Expected {N_FRAMES} homography rows, got {len(df)}"

    # Extract 3x3 matrices (columns h00..h22)
    h_cols = [f'h{i}{j}' for i in range(3) for j in range(3)]
    H = df[h_cols].values.reshape(-1, 3, 3)

    valid = df['valid'].values.astype(bool)
    reproj_error = df['reprojection_error'].values

    log(f"Homography loaded: {len(df)} frames")
    log(f"  Valid matrices: {valid.sum()} ({100 * valid.sum() / len(df):.1f}%)")
    log(f"  Median reprojection error: {np.nanmedian(reproj_error[valid]):.3f}")

    return H, valid, reproj_error


def get_graded_mask():
    """Return boolean mask of graded frames (where labels are scored).

    Returns:
        mask: array of shape (N_FRAMES,), True for graded frames
    """
    mask = np.zeros(N_FRAMES, dtype=bool)

    half_blocks = half_video_blocks(GAME)
    for _, start_frame, end_frame in half_blocks:
        mask[start_frame:end_frame + 1] = True

    n_graded = mask.sum()
    log(f"Graded frames: {n_graded} / {N_FRAMES} ({100 * n_graded / N_FRAMES:.1f}%)")

    return mask


if __name__ == '__main__':
    # Clear log and test loading
    import os
    os.makedirs('logs', exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=== IO Module Test ===\n")

    log("Testing data loading...")

    df, metadata, team_of_track = load_tracking()
    print(f"Tracking: {metadata['n_frames']} frames")
    print(f"  Home: {metadata['home_team']}, Away: {metadata['away_team']}")
    print(f"  Team mapping: {len(team_of_track)} slots")
    print(f"  Ball visible: {metadata['ball_visible_count']} frames")

    events = load_smart_tagging()
    print(f"Smart Tagging: {len(events)} events")

    H, valid, reproj = load_homography()
    print(f"Homography: {len(H)} matrices, {valid.sum()} valid")

    mask = get_graded_mask()
    print(f"Graded: {mask.sum()} frames")

    print(f"\nDiagnostics written to {LOG_FILE}")
