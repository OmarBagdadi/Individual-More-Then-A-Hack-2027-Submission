"""Write submission CSV file.

Ensures exact format compliance with validation requirements.
"""

import pandas as pd
import os
from src.possession.config import N_FRAMES, LOG_FILE


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def write_submission(labels, output_path):
    """Write labels to submission CSV.

    Args:
        labels: array of labels ('H', 'A', 'D', 'X')
        output_path: path to output CSV file

    Returns:
        output_path
    """
    assert len(labels) == N_FRAMES, f"Expected {N_FRAMES} labels, got {len(labels)}"

    # Create DataFrame
    df = pd.DataFrame({
        'frame': range(N_FRAMES),
        'label': labels
    })

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write CSV
    df.to_csv(output_path, index=False)

    # Validate
    assert os.path.exists(output_path), f"Failed to write {output_path}"

    # Count labels
    counts = df['label'].value_counts().to_dict()

    log(f"\nSubmission written: {output_path}")
    log(f"  H: {counts.get('H', 0)} frames")
    log(f"  A: {counts.get('A', 0)} frames")
    log(f"  D: {counts.get('D', 0)} frames")
    log(f"  X: {counts.get('X', 0)} frames")
    log(f"  Total: {len(labels)} frames")

    return output_path


if __name__ == '__main__':
    import numpy as np

    # Test with dummy labels
    labels = np.random.choice(['H', 'A', 'D', 'X'], N_FRAMES)
    output_path = write_submission(labels, 'evaluation/results/test.csv')
    print(f"Test submission written to {output_path}")
