"""UF-27: Streaming Backprojection Imaging.

Implements tiled BP with overlap+stitch for memory-bounded execution.
NumPy-based with vectorized pulse batch processing.
"""
import numpy as np
import time
from ..data_contracts import PlatformMeta, RadarMeta, BPGrid
import logging

logger = logging.getLogger(__name__)

C = 299792458.0


def _bp_kernel_batch(raw_batch: np.ndarray,
                     positions_batch: np.ndarray,
                     grid_xyz: np.ndarray,
                     tau_min: float, dtau: float, Nr: int,
                     fc: float) -> np.ndarray:
    """Vectorized BP kernel for a batch of pulses.

    Args:
        raw_batch: [Kb, Nr] echo data for this batch.
        positions_batch: [Kb, 3] platform positions.
        grid_xyz: [M, 3] pixel coordinates.
        tau_min, dtau, Nr: range axis params.
        fc: carrier frequency.

    Returns:
        contribution: [M] complex accumulation.
    """
    Kb = raw_batch.shape[0]
    M = grid_xyz.shape[0]

    result = np.zeros(M, dtype=np.complex128)

    for k in range(Kb):
        diff = grid_xyz - positions_batch[k, :]  # [M, 3]
        R = np.sqrt(np.sum(diff ** 2, axis=1))  # [M]
        tau = 2.0 * R / C

        idx_float = (tau - tau_min) / dtau
        idx0 = np.floor(idx_float).astype(np.int64)
        frac = idx_float - idx0

        valid = (idx0 >= 0) & (idx0 < Nr - 1)
        v = np.where(valid)[0]
        i0 = idx0[v]
        f = frac[v]

        echo_interp = np.zeros(M, dtype=np.complex128)
        echo_interp[v] = raw_batch[k, i0] * (1 - f) + raw_batch[k, i0 + 1] * f

        phase = 4.0 * np.pi * fc * R / C
        result += echo_interp * np.exp(1j * phase)

    return result


def backproject_full(raw: np.ndarray,
                     tau_axis: np.ndarray,
                     platform: PlatformMeta,
                     radar: RadarMeta,
                     bp_grid: BPGrid) -> np.ndarray:
    """Full (non-tiled) backprojection.

    I(x) = sum_k s(k, tau(x,k)) * exp(+j * 4*pi*fc * R(x,k) / c)

    Returns:
        image: [Ny, Nx] complex64 SLC image.
    """
    K = platform.K
    Nx = bp_grid.Nx
    Ny = bp_grid.Ny
    fc = radar.fc

    grid_xyz = bp_grid.get_grid_xyz()  # [M, 3]
    M = grid_xyz.shape[0]

    t_start = time.time()
    logger.info(f"BP full: M={M} pixels, K={K} pulses")

    tau_min = tau_axis[0]
    Nr = len(tau_axis)
    dtau = (tau_axis[-1] - tau_axis[0]) / (Nr - 1) if Nr > 1 else 1.0

    image_flat = _bp_kernel_batch(raw, platform.positions, grid_xyz,
                                  tau_min, dtau, Nr, fc)

    elapsed = time.time() - t_start
    logger.info(f"BP full completed: {elapsed:.2f}s, "
                f"{M*K/elapsed/1e6:.1f} Mpixel-pulses/s")

    return image_flat.reshape(Ny, Nx).astype(np.complex64)


def backproject_tiled(raw: np.ndarray,
                      tau_axis: np.ndarray,
                      platform: PlatformMeta,
                      radar: RadarMeta,
                      bp_grid: BPGrid,
                      tile_size: tuple = (128, 128),
                      k_batch: int = 64,
                      overlap: int = 16,
                      stitch_method: str = 'linear_blend') -> np.ndarray:
    """Tiled backprojection with overlap and stitching.

    Args:
        raw: [K, Nr] range-compressed echo.
        tau_axis: [Nr] range time axis.
        platform: Platform state vectors.
        radar: Radar parameters.
        bp_grid: Output grid.
        tile_size: (tile_nx, tile_ny) in pixels.
        k_batch: Pulses per batch.
        overlap: Overlap in pixels between tiles.
        stitch_method: 'linear_blend' or 'replace'.

    Returns:
        image: [Ny, Nx] complex64 SLC image.
    """
    K = platform.K
    Nx = bp_grid.Nx
    Ny = bp_grid.Ny
    fc = radar.fc

    tile_nx, tile_ny = tile_size
    t_start = time.time()

    # Count tiles
    step_x = max(tile_nx - overlap, 1)
    step_y = max(tile_ny - overlap, 1)
    n_tiles_x = (Nx + step_x - 1) // step_x
    n_tiles_y = (Ny + step_y - 1) // step_y
    total_tiles = n_tiles_x * n_tiles_y

    logger.info(f"BP tiled: {Nx}x{Ny} image, tile={tile_nx}x{tile_ny}, "
                f"overlap={overlap}, k_batch={k_batch}, total_tiles={total_tiles}")

    # Output image and weight map for blending
    image = np.zeros((Ny, Nx), dtype=np.complex128)
    weight = np.zeros((Ny, Nx), dtype=np.float64)

    # Range axis parameters
    tau_min = tau_axis[0]
    Nr = len(tau_axis)
    dtau = (tau_axis[-1] - tau_axis[0]) / (Nr - 1) if Nr > 1 else 1.0

    n_done = 0
    for ty_start in range(0, Ny, step_y):
        ty_end = min(ty_start + tile_ny, Ny)
        for tx_start in range(0, Nx, step_x):
            tx_end = min(tx_start + tile_nx, Nx)
            n_done += 1

            # Extract tile grid
            tile_x = bp_grid.x_axis[tx_start:tx_end]
            tile_y = bp_grid.y_axis[ty_start:ty_end]
            txx, tyy = np.meshgrid(tile_x, tile_y)
            tile_z = np.full_like(txx, bp_grid.z_val)
            tile_xyz = np.column_stack([
                txx.ravel(), tyy.ravel(), tile_z.ravel()
            ])  # [Mt, 3]

            tile_image = np.zeros(tile_xyz.shape[0], dtype=np.complex128)

            # Process in K batches
            for kb_start in range(0, K, k_batch):
                kb_end = min(kb_start + k_batch, K)
                tile_image += _bp_kernel_batch(
                    raw[kb_start:kb_end], platform.positions[kb_start:kb_end],
                    tile_xyz, tau_min, dtau, Nr, fc)

            tile_img_2d = tile_image.reshape(ty_end - ty_start, tx_end - tx_start)

            # Blending weight
            tile_weight = _compute_blend_weight(
                ty_end - ty_start, tx_end - tx_start, overlap,
                ty_start == 0, tx_start == 0,
                ty_end == Ny, tx_end == Nx)

            image[ty_start:ty_end, tx_start:tx_end] += tile_img_2d * tile_weight
            weight[ty_start:ty_end, tx_start:tx_end] += tile_weight

            if n_done % max(1, total_tiles // 5) == 0 or n_done == total_tiles:
                elapsed = time.time() - t_start
                logger.info(f"  Tile {n_done}/{total_tiles} done "
                            f"({100*n_done/total_tiles:.0f}%, {elapsed:.1f}s)")

    # Normalize by weight
    nonzero = weight > 1e-12
    image[nonzero] /= weight[nonzero]

    elapsed = time.time() - t_start
    total_mpps = Nx * Ny * K / elapsed / 1e6
    logger.info(f"BP tiled completed: {n_done} tiles, {elapsed:.2f}s, "
                f"{total_mpps:.1f} Mpixel-pulses/s")

    return image.astype(np.complex64)


def _compute_blend_weight(ny: int, nx: int, overlap: int,
                          is_top: bool, is_left: bool,
                          is_bottom: bool, is_right: bool) -> np.ndarray:
    """Compute linear blend weight for a tile.

    Interior pixels get weight=1, overlap edges get linear ramp [0,1].
    """
    wy = np.ones(ny, dtype=np.float64)
    wx = np.ones(nx, dtype=np.float64)

    if overlap > 0:
        ramp = np.linspace(0, 1, overlap + 1)[1:]
        if not is_top and overlap <= ny:
            wy[:overlap] = ramp
        if not is_bottom and overlap <= ny:
            wy[-overlap:] = ramp[::-1]
        if not is_left and overlap <= nx:
            wx[:overlap] = ramp
        if not is_right and overlap <= nx:
            wx[-overlap:] = ramp[::-1]

    return wy[:, np.newaxis] * wx[np.newaxis, :]
