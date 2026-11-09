"""Ball reconstruction: interpolate short gaps, model uncertainty for long gaps.

Handles the 41% missing ball problem with confidence-aware reconstruction.
"""

import numpy as np
from scipy import interpolate
from src.possession.config import (
    BALL_SHORT_GAP_FRAMES,
    BALL_UNCERTAINTY_GROWTH_RATE,
    BALL_PROCESS_NOISE,
    BALL_MEASUREMENT_NOISE,
    PITCH_LENGTH,
    PITCH_WIDTH,
    FPS,
    LOG_FILE
)


def log(message):
    """Append message to log file."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


""" 1D Kalman filter for position and velocity estimation.

    Args:
        observations: array of observations (NaN for missing)
        missing_mask: boolean array, True where observations are missing
        process_noise: process noise variance
        measurement_noise: measurement noise variance

    Returns:
        positions: filtered positions
        velocities: estimated velocities
        uncertainties: position uncertainties
"""
def simple_kalman_1d(observations, missing_mask, process_noise, measurement_noise):
    n = len(observations)
    dt = 1.0 / FPS  # time step in seconds

    # State: [position, velocity]
    positions = np.full(n, np.nan)
    velocities = np.full(n, np.nan)
    uncertainties = np.full(n, np.nan)

    # Initialize with first valid observation
    valid_indices = np.where(~missing_mask)[0]
    if len(valid_indices) == 0:
        return positions, velocities, uncertainties

    # State estimate and covariance
    x = np.array([observations[valid_indices[0]], 0.0])  # position, velocity
    P = np.array([[measurement_noise, 0], [0, 1.0]])  # initial covariance

    # State transition matrix (constant velocity model)
    F = np.array([[1, dt], [0, 1]])

    # Observation matrix (we observe position only)
    H = np.array([[1, 0]])

    # Process noise covariance
    Q = process_noise * np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])

    # Measurement noise covariance
    R = np.array([[measurement_noise]])

    for i in range(n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update if observation available
        if not missing_mask[i]:
            y = observations[i] - H @ x  # innovation
            S = H @ P @ H.T + R  # innovation covariance
            K = P @ H.T / S  # Kalman gain
            x = x + K * y
            P = (np.eye(2) - K @ H) @ P

        positions[i] = x[0]
        velocities[i] = x[1]
        uncertainties[i] = np.sqrt(P[0, 0])

    return positions, velocities, uncertainties


""" Reconstruct ball trajectory with confidence estimates.

    Strategy:
    - Short gaps (≤ BALL_SHORT_GAP_FRAMES): cubic interpolation
    - Long gaps: Kalman filter with growing uncertainty
    - Output confidence: 1.0 for observed, decaying for interpolated/extrapolated

    Args:
        ball_x: array of ball x positions (metres, NaN for missing)
        ball_y: array of ball y positions (metres, NaN for missing)

    Returns:
        dict with keys:
            x: reconstructed x positions (metres)
            y: reconstructed y positions (metres)
            speed: instantaneous speed (m/s)
            confidence: confidence in position (0-1)
            visible: original visibility mask
"""
def reconstruct_ball(ball_x, ball_y):
    n_frames = len(ball_x)
    visible = ~(np.isnan(ball_x) | np.isnan(ball_y))

    # Initialize outputs
    recon_x = ball_x.copy()
    recon_y = ball_y.copy()
    confidence = visible.astype(float)

    # Identify gaps
    gap_starts = []
    gap_ends = []
    in_gap = False

    for i in range(n_frames):
        if not visible[i] and not in_gap:
            gap_starts.append(i)
            in_gap = True
        elif visible[i] and in_gap:
            gap_ends.append(i - 1)
            in_gap = False

    if in_gap:  # Gap extends to end
        gap_ends.append(n_frames - 1)

    # Process each gap
    n_short_gaps = 0
    n_long_gaps = 0
    total_short_frames = 0
    total_long_frames = 0

    for gap_start, gap_end in zip(gap_starts, gap_ends):
        gap_length = gap_end - gap_start + 1

        if gap_length <= BALL_SHORT_GAP_FRAMES:
            # Short gap: cubic interpolation
            n_short_gaps += 1
            total_short_frames += gap_length

            # Find surrounding valid points
            before = gap_start - 1
            after = gap_end + 1

            # Need at least 2 points on each side for cubic, fall back to linear if not available
            before_indices = []
            after_indices = []

            for j in range(max(0, before - 3), before + 1):
                if j >= 0 and visible[j]:
                    before_indices.append(j)

            for j in range(after, min(n_frames, after + 4)):
                if j < n_frames and visible[j]:
                    after_indices.append(j)

            all_indices = before_indices + after_indices

            if len(all_indices) >= 2:
                # Interpolate
                interp_indices = np.array(all_indices)
                gap_indices = np.arange(gap_start, gap_end + 1)

                # Use linear for < 4 points, cubic otherwise
                kind = 'cubic' if len(all_indices) >= 4 else 'linear'

                try:
                    f_x = interpolate.interp1d(interp_indices, ball_x[interp_indices],
                                               kind=kind, fill_value='extrapolate')
                    f_y = interpolate.interp1d(interp_indices, ball_y[interp_indices],
                                               kind=kind, fill_value='extrapolate')

                    recon_x[gap_indices] = f_x(gap_indices)
                    recon_y[gap_indices] = f_y(gap_indices)

                    # Confidence decays linearly within the gap
                    mid = (gap_start + gap_end) / 2
                    for i in gap_indices:
                        dist_from_edge = min(abs(i - gap_start), abs(i - gap_end))
                        confidence[i] = max(0.5, 1.0 - (gap_length - dist_from_edge) / gap_length * 0.5)

                except Exception:
                    # Interpolation failed, use forward fill
                    if before >= 0:
                        recon_x[gap_indices] = ball_x[before]
                        recon_y[gap_indices] = ball_y[before]
                        confidence[gap_indices] = 0.3
            else:
                # No surrounding points, forward/backward fill
                if before >= 0 and visible[before]:
                    recon_x[gap_indices] = ball_x[before]
                    recon_y[gap_indices] = ball_y[before]
                    confidence[gap_indices] = 0.2
                elif after < n_frames and visible[after]:
                    recon_x[gap_indices] = ball_x[after]
                    recon_y[gap_indices] = ball_y[after]
                    confidence[gap_indices] = 0.2

        else:
            # Long gap: use Kalman filter with uncertainty tracking
            n_long_gaps += 1
            total_long_frames += gap_length

            # For very long gaps, just mark as low confidence and use last known position
            before = gap_start - 1
            after = gap_end + 1

            if before >= 0 and visible[before]:
                # Forward fill with decaying confidence
                gap_indices = np.arange(gap_start, gap_end + 1)
                recon_x[gap_indices] = ball_x[before]
                recon_y[gap_indices] = ball_y[before]

                # Confidence decays exponentially
                for i, idx in enumerate(gap_indices):
                    confidence[idx] = np.exp(-BALL_UNCERTAINTY_GROWTH_RATE * i)
            elif after < n_frames and visible[after]:
                # Backward fill
                gap_indices = np.arange(gap_start, gap_end + 1)
                recon_x[gap_indices] = ball_x[after]
                recon_y[gap_indices] = ball_y[after]
                confidence[gap_indices] = np.exp(-BALL_UNCERTAINTY_GROWTH_RATE * gap_length / 2)

    # Compute speed from reconstructed positions
    speed = np.zeros(n_frames)
    for i in range(1, n_frames):
        dx = recon_x[i] - recon_x[i-1]
        dy = recon_y[i] - recon_y[i-1]
        speed[i] = np.sqrt(dx*dx + dy*dy) * FPS  # convert to m/s

    # Smooth speed slightly to avoid spikes
    from scipy.ndimage import gaussian_filter1d
    speed = gaussian_filter1d(speed, sigma=1.0)

    # Statistics
    n_visible = visible.sum()
    n_reconstructed = (~np.isnan(recon_x)).sum()

    log(f"\nBall reconstruction:")
    log(f"  Visible: {n_visible} frames ({100*n_visible/n_frames:.1f}%)")
    log(f"  Reconstructed: {n_reconstructed} frames ({100*n_reconstructed/n_frames:.1f}%)")
    log(f"  Short gaps (<={BALL_SHORT_GAP_FRAMES} frames): {n_short_gaps} gaps, {total_short_frames} frames")
    log(f"  Long gaps (>{BALL_SHORT_GAP_FRAMES} frames): {n_long_gaps} gaps, {total_long_frames} frames")
    log(f"  Mean confidence: {confidence.mean():.3f}")
    log(f"  Mean speed: {speed.mean():.2f} m/s, max: {speed.max():.2f} m/s")

    return {
        'x': recon_x,
        'y': recon_y,
        'speed': speed,
        'confidence': confidence,
        'visible': visible,
    }


if __name__ == '__main__':
    # Test with synthetic data
    import sys
    sys.path.insert(0, '.')

    from src.possession.io import load_tracking

    print("Testing ball reconstruction...")
    df, metadata = load_tracking()

    ball_x = df['ball_x'].values
    ball_y = df['ball_y'].values

    result = reconstruct_ball(ball_x, ball_y)

    print(f"Original visible: {result['visible'].sum()} frames")
    print(f"Reconstructed: {(~np.isnan(result['x'])).sum()} frames")
    print(f"Mean confidence: {result['confidence'].mean():.3f}")
    print(f"Mean speed: {result['speed'].mean():.2f} m/s")
