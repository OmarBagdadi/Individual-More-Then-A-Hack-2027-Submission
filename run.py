"""Main entry point for possession labelling pipeline.

Usage:
    python run.py

Generates submission CSV deterministically from provided data.
"""

"""
Ensure the correct Python version is used. The pipeline requires at least Python 3.9 and at most Python 3.12,
and will exit with an error message if a different version is detected.
"""
import sys
if sys.version_info[:2] < (3, 9) or sys.version_info[:2] > (3, 12):
    sys.exit(f"Python 3.9+ up till 3.12 required, found {sys.version.split()[0]}. See README.")

import os
import sys
import numpy as np
import time

# Ensure src is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.possession.config import LOG_FILE, RANDOM_SEED
from src.possession.io import load_tracking, load_smart_tagging, load_homography, get_graded_mask, log
from src.possession.ball import reconstruct_ball
from src.possession.evidence import compute_control_evidence
from src.possession.dead import compute_dead_ball_probability
from src.possession.decode import viterbi_decode
from src.possession.weak_eval import weak_eval
from src.possession.submission import write_submission

# Set random seed
np.random.seed(RANDOM_SEED)

def main():
    """Run the full possession labelling pipeline."""
    start_time = time.time()

    # Clear log
    os.makedirs('logs', exist_ok=True)
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("=== Possession Labelling Pipeline ===\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("Stage 1: Loading data...")
    log("\n=== Stage 1: Loading Data ===")

    tracking_df, metadata, team_of_track = load_tracking()
    events = load_smart_tagging()
    H_matrices, valid, reproj = load_homography()
    graded_mask = get_graded_mask()

    print(f"  OK - Loaded {metadata['n_frames']} frames")

    # Stage 2: Ball reconstruction
    print("\nStage 2: Ball reconstruction...")
    log("\n=== Stage 2: Ball Reconstruction ===")

    ball_x = tracking_df['ball_x'].values
    ball_y = tracking_df['ball_y'].values
    ball_state = reconstruct_ball(ball_x, ball_y)

    print(f"  OK - Reconstructed {(~np.isnan(ball_state['x'])).sum()} frames")

    # Stage 3: Evidence accumulation
    print("\nStage 3: Evidence accumulation...")
    log("\n=== Stage 3: Evidence Accumulation ===")

    evidence = compute_control_evidence(tracking_df, ball_state, metadata, team_of_track)

    print(f"  OK - Computed control scores")

    # Stage 4: Dead ball detection
    print("\nStage 4: Dead ball detection...")
    log("\n=== Stage 4: Dead Ball Detection ===")

    dead_prob = compute_dead_ball_probability(tracking_df, ball_state, events)

    print(f"  OK - Computed dead ball probabilities")

    # Stage 5: Viterbi decoding
    print("\nStage 5: Viterbi decoding...")
    log("\n=== Stage 5: Viterbi Decoding ===")

    labels = viterbi_decode(evidence, dead_prob, ball_state, events, graded_mask)

    print(f"  OK - Decoded possession sequence")

    # Stage 6: Weak evaluation
    print("\nStage 6: Weak evaluation...")
    log("\n=== Stage 6: Weak Evaluation ===")

    metrics = weak_eval(labels, events)

    print(f"  OK - Team agreement: {metrics['team_agreement']:.3f}")
    print(f"  OK - Possessions: {metrics['n_possessions']}")

    # Write submission
    output_path = 'evaluation/results/submission.csv'
    write_submission(labels, output_path)

    elapsed = time.time() - start_time

    log(f"\n=== Pipeline Complete ===")
    log(f"Elapsed time: {elapsed:.1f} seconds")

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f} seconds")
    print(f"Submission written to: {output_path}")
    print(f"Diagnostics written to: {LOG_FILE}")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Validate: python evaluation/validate_submission.py {output_path}")
    print(f"  2. Score:    python evaluation/run_evaluation.py")

if __name__ == '__main__':
    main()
