"""Synthetic platform trajectory generation.

Generates stripmap-mode platform state vectors (SR-04b).
"""
import numpy as np
from ..data_contracts import PlatformMeta
import logging

logger = logging.getLogger(__name__)


def generate_stripmap_trajectory(
        altitude: float = 5000.0,
        velocity: float = 100.0,
        look_angle_deg: float = 45.0,
        azimuth_deg: float = 0.0,
        num_pulses: int = 256,
        prf: float = 500.0,
        scene_center: np.ndarray = None,
) -> PlatformMeta:
    """Generate a synthetic straight-line (stripmap) trajectory.

    The platform flies along the azimuth direction at constant altitude.
    The look direction is perpendicular to the flight path (side-looking).

    Args:
        altitude: Flight altitude above ground [m].
        velocity: Platform speed [m/s].
        look_angle_deg: Look angle from nadir [deg].
        azimuth_deg: Flight direction from North [deg]. 0=North(+y), 90=East(+x).
        num_pulses: Number of pulses (K).
        prf: Pulse repetition frequency [Hz].
        scene_center: [3] Scene center position [m]. None = [0,0,0].

    Returns:
        PlatformMeta with state vectors.
    """
    if scene_center is None:
        scene_center = np.array([0.0, 0.0, 0.0])

    K = num_pulses
    dt = 1.0 / prf
    t_k = np.arange(K, dtype=np.float64) * dt

    # Flight direction unit vector (in ENU x-y plane)
    az_rad = np.radians(azimuth_deg)
    flight_dir = np.array([np.sin(az_rad), np.cos(az_rad), 0.0])

    # Total aperture length
    aperture_length = velocity * (K - 1) * dt
    start_offset = -aperture_length / 2

    # Look direction (perpendicular to flight, pointing right)
    # For side-looking radar, offset the platform from scene center
    look_dir = np.array([np.cos(az_rad), -np.sin(az_rad), 0.0])  # perpendicular to flight
    ground_range = altitude * np.tan(np.radians(look_angle_deg))

    # Platform start position
    start_pos = scene_center.copy()
    start_pos[2] = altitude
    start_pos[:2] -= look_dir[:2] * ground_range  # offset from scene center
    start_pos[:2] += flight_dir[:2] * start_offset  # center aperture on scene

    positions = np.zeros((K, 3), dtype=np.float64)
    velocities = np.zeros((K, 3), dtype=np.float64)
    for k in range(K):
        positions[k, :] = start_pos + flight_dir * velocity * t_k[k]
        velocities[k, :] = flight_dir * velocity

    platform = PlatformMeta(t_k=t_k, positions=positions, velocities=velocities)
    platform.validate()

    slant_range = np.sqrt(ground_range**2 + altitude**2)
    logger.info(f"Platform trajectory: K={K}, altitude={altitude:.0f} m, "
                f"velocity={velocity:.0f} m/s, ground_range={ground_range:.0f} m, "
                f"slant_range={slant_range:.0f} m, aperture={aperture_length:.1f} m")

    return platform
