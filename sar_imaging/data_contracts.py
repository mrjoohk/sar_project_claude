"""Data contracts and validators for the SAR pipeline.

Enforces the data contracts defined in 02_unit_functions_urban_sar.md.
All data structures use numpy arrays (SoA layout, GPU-friendly).
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RadarMeta:
    """Radar parameters."""
    fc: float          # Center frequency [Hz]
    bandwidth: float   # Bandwidth [Hz]
    prf: float         # Pulse repetition frequency [Hz]
    tp: float          # Pulse duration [s]
    c: float = 299792458.0  # Speed of light [m/s]

    @property
    def wavelength(self) -> float:
        return self.c / self.fc

    @property
    def range_resolution(self) -> float:
        return self.c / (2 * self.bandwidth)

    def validate(self):
        assert self.fc > 0, f"fc must be positive, got {self.fc}"
        assert self.bandwidth > 0, f"bandwidth must be positive, got {self.bandwidth}"
        assert self.prf > 0, f"prf must be positive, got {self.prf}"
        assert self.tp > 0, f"tp must be positive, got {self.tp}"
        assert self.bandwidth <= self.fc, f"bandwidth ({self.bandwidth}) > fc ({self.fc})"


@dataclass
class PlatformMeta:
    """Platform state vectors.

    All arrays indexed by pulse index k = 0..K-1.
    """
    t_k: np.ndarray          # [K] Time tags [s], float64
    positions: np.ndarray     # [K, 3] Platform positions (x,y,z) in ENU [m], float64
    velocities: np.ndarray    # [K, 3] Platform velocities [m/s], float64

    def validate(self):
        K = self.t_k.shape[0]
        assert self.t_k.ndim == 1, f"t_k must be 1D, got {self.t_k.ndim}D"
        assert self.positions.shape == (K, 3), \
            f"positions shape {self.positions.shape} != ({K}, 3)"
        assert self.velocities.shape == (K, 3), \
            f"velocities shape {self.velocities.shape} != ({K}, 3)"
        assert np.all(np.diff(self.t_k) > 0), "t_k must be strictly increasing"

    @property
    def K(self) -> int:
        return self.t_k.shape[0]


@dataclass
class Scatterers:
    """Scatterer list in Structure-of-Arrays layout.

    Required: sx, sy, sz, s_rcs (all float32).
    Optional: normals, areas, vis_mask, layover_flag.
    """
    sx: np.ndarray    # [N] x positions [m], float32
    sy: np.ndarray    # [N] y positions [m], float32
    sz: np.ndarray    # [N] z positions [m], float32
    s_rcs: np.ndarray # [N] RCS values, float32

    # Optional fields
    snx: Optional[np.ndarray] = None  # [N] normal x
    sny: Optional[np.ndarray] = None  # [N] normal y
    snz: Optional[np.ndarray] = None  # [N] normal z
    s_area: Optional[np.ndarray] = None  # [N] facet area
    vis_mask: Optional[np.ndarray] = None  # [N] visibility mask [0,1]
    layover_flag: Optional[np.ndarray] = None  # [N] layover flag

    def validate(self):
        N = self.sx.shape[0]
        assert self.sx.ndim == 1, f"sx must be 1D, got {self.sx.ndim}D"
        assert self.sy.shape == (N,), f"sy shape mismatch: {self.sy.shape} != ({N},)"
        assert self.sz.shape == (N,), f"sz shape mismatch: {self.sz.shape} != ({N},)"
        assert self.s_rcs.shape == (N,), f"s_rcs shape mismatch: {self.s_rcs.shape} != ({N},)"
        if self.vis_mask is not None:
            assert self.vis_mask.shape == (N,)
        if self.layover_flag is not None:
            assert self.layover_flag.shape == (N,)

    @property
    def N(self) -> int:
        return self.sx.shape[0]

    @property
    def positions(self) -> np.ndarray:
        """Return [N, 3] position array."""
        return np.column_stack([self.sx, self.sy, self.sz])

    def get_effective_rcs(self) -> np.ndarray:
        """Return RCS with visibility mask applied."""
        rcs = self.s_rcs.copy()
        if self.vis_mask is not None:
            rcs *= self.vis_mask
        return rcs


@dataclass
class BPGrid:
    """Backprojection output pixel grid.

    Defines a 2D grid in ENU coordinates at a fixed height (z=0 default).
    """
    x_axis: np.ndarray  # [Nx] x coordinates [m]
    y_axis: np.ndarray  # [Ny] y coordinates [m]
    z_val: float = 0.0  # Fixed height [m]

    def validate(self):
        assert self.x_axis.ndim == 1
        assert self.y_axis.ndim == 1
        assert len(self.x_axis) >= 2
        assert len(self.y_axis) >= 2

    @property
    def Nx(self) -> int:
        return len(self.x_axis)

    @property
    def Ny(self) -> int:
        return len(self.y_axis)

    @property
    def M(self) -> int:
        return self.Nx * self.Ny

    @property
    def spacing_x(self) -> float:
        return float(self.x_axis[1] - self.x_axis[0])

    @property
    def spacing_y(self) -> float:
        return float(self.y_axis[1] - self.y_axis[0])

    def get_grid_xyz(self) -> np.ndarray:
        """Return [M, 3] array of grid point coordinates."""
        xx, yy = np.meshgrid(self.x_axis, self.y_axis, indexing='xy')
        zz = np.full_like(xx, self.z_val)
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def validate_raw(raw: np.ndarray, K: int):
    """Validate raw echo data."""
    assert raw.ndim == 2, f"raw must be 2D, got {raw.ndim}D"
    assert raw.shape[0] == K, f"raw pulse count {raw.shape[0]} != K={K}"
    assert np.iscomplexobj(raw), "raw must be complex"
    energy = np.sum(np.abs(raw) ** 2)
    if energy < 1e-30:
        raise ValueError(f"Raw echo energy is near zero ({energy:.2e}). Signal collapse detected.")
