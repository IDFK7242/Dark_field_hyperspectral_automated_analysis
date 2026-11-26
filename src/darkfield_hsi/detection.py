"""Stage 1: Detection-only TIFF cleaning, LoG detection, spot morphology, and drift-registered tracking."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import blob_log
from skimage.registration import phase_cross_correlation

from darkfield_hsi.config import DetectionConfig, TrackingConfig
from darkfield_hsi.io_utils import FrameInput

logger = logging.getLogger(__name__)

BLOB_COLUMNS = [
    "Blob_ID",
    "Frame_Number",
    "X",
    "Y",
    "Radius",
    "Spot_Peak_Intensity",
    "Spot_Area_px",
    "Spot_Radius_FWHM_px",
    "Eccentricity",
    "Is_Saturated",
    "Filename",
]

DRIFT_COLUMNS = ["Frame_Number", "Filename", "Shift_Y_px", "Shift_X_px", "Registration_Error"]


def project_channels(img_bgr: np.ndarray, projection: str) -> np.ndarray:
    """Convert BGR or grayscale image to a 2D float32 luminance array in [0, 255]."""
    if img_bgr.ndim == 2:
        raw = img_bgr
    elif projection == "max_rgb":
        raw = img_bgr[..., :3].max(axis=2)
    elif projection == "mean_rgb":
        raw = img_bgr[..., :3].astype(np.float32).mean(axis=2)
    elif projection == "gray":
        raw = cv2.cvtColor(img_bgr[..., :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported channel_projection: {projection!r}")

    if raw.dtype == np.uint16:
        return (raw.astype(np.float32) / 257.0)
    return raw.astype(np.float32)


def prepare_detection_image(
    img_bgr: np.ndarray,
    cfg: DetectionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce (cleaned_0_1, destriped_raw_255) for detection and morphology measurement.

    The original RGB TIFF is never modified on disk; cleaning is applied only to the
    in-memory detection copy to remove push-broom striping and shot noise.
    """
    proj = project_channels(img_bgr, cfg.channel_projection)

    if not cfg.clean_enabled:
        return np.clip(proj / 255.0, 0.0, 1.0), proj

    work = proj.copy()
    if cfg.destripe_columns:
        col_med = np.median(work, axis=0)
        work = np.clip(work - col_med[None, :], 0.0, 255.0)

    destriped_raw = work.copy()

    if cfg.median_ksize >= 3:
        work_u8 = np.clip(np.rint(work), 0, 255).astype(np.uint8)
        work = cv2.medianBlur(work_u8, cfg.median_ksize).astype(np.float32)

    if cfg.gaussian_sigma > 0:
        work = cv2.GaussianBlur(work, (0, 0), sigmaX=cfg.gaussian_sigma, sigmaY=cfg.gaussian_sigma)

    if cfg.bg_clip_mad_mult > 0:
        med = float(np.median(work))
        mad = float(np.median(np.abs(work - med)))
        floor = med + cfg.bg_clip_mad_mult * max(mad, 1.0)
        work = np.clip(work - floor, 0.0, 255.0)

    cleaned = np.clip(work / 255.0, 0.0, 1.0)
    return cleaned, destriped_raw


def detect_blobs(image_for_log: np.ndarray, cfg: DetectionConfig) -> np.ndarray:
    """Detect bright blobs in a 2D image using LoG.

    Returns an array of shape (N, 3) with columns [y, x, radius].
    """
    blobs = blob_log(
        image_for_log,
        max_sigma=cfg.max_sigma,
        num_sigma=cfg.num_sigma,
        threshold=cfg.threshold,
        overlap=cfg.overlap,
    )
    if blobs.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    blobs = blobs.astype(np.float64, copy=True)
    blobs[:, 2] = blobs[:, 2] * cfg.radius_scale
    blobs = blobs[blobs[:, 2] > cfg.min_radius]

    if cfg.border_margin_px > 0 and len(blobs) > 0:
        height, width = image_for_log.shape[:2]
        m = float(cfg.border_margin_px)
        keep = (
            (blobs[:, 0] >= m)
            & (blobs[:, 0] < height - m)
            & (blobs[:, 1] >= m)
            & (blobs[:, 1] < width - m)
        )
        blobs = blobs[keep]

    return blobs


def measure_spot_morphology(
    raw_proj: np.ndarray,
    x: float,
    y: float,
    patch_half: int = 10,
) -> dict[str, float | int | bool]:
    """Measure local peak intensity, FWHM area/radius, eccentricity, and saturation."""
    height, width = raw_proj.shape[:2]
    ix = int(np.clip(round(x), 0, width - 1))
    iy = int(np.clip(round(y), 0, height - 1))

    x0 = max(0, ix - patch_half)
    x1 = min(width, ix + patch_half + 1)
    y0 = max(0, iy - patch_half)
    y1 = min(height, iy + patch_half + 1)
    patch = raw_proj[y0:y1, x0:x1].astype(np.float64)

    cx_loc = ix - x0
    cy_loc = iy - y0

    # Local background from outer 2-pixel border of the patch
    border_mask = np.zeros(patch.shape, dtype=bool)
    border_mask[:2, :] = True
    border_mask[-2:, :] = True
    border_mask[:, :2] = True
    border_mask[:, -2:] = True
    border_vals = patch[border_mask]
    local_bg = float(np.median(border_vals)) if border_vals.size else 0.0
    local_mad = float(np.median(np.abs(border_vals - local_bg))) if border_vals.size else 0.0

    # Peak intensity in a 3x3 neighborhood centered at detected (x, y)
    py0, py1 = max(0, cy_loc - 1), min(patch.shape[0], cy_loc + 2)
    px0, px1 = max(0, cx_loc - 1), min(patch.shape[1], cx_loc + 2)
    core = patch[py0:py1, px0:px1]
    peak_flat = int(np.argmax(core))
    peak_dy, peak_dx = divmod(peak_flat, core.shape[1])
    pk_y = py0 + peak_dy
    pk_x = px0 + peak_dx
    spot_peak = float(patch[pk_y, pk_x])
    is_saturated = bool(spot_peak >= 254.0)

    # Half-maximum foreground mask restricted to the connected component containing the peak
    above_bg = max(0.0, spot_peak - local_bg)
    thresh = local_bg + max(3.0 * max(local_mad, 1.0), 0.5 * above_bg)
    fg = (patch >= thresh).astype(np.uint8)
    fg[pk_y, pk_x] = 1

    _, labels = cv2.connectedComponents(fg, connectivity=8)
    target_label = labels[pk_y, pk_x]
    comp_mask = labels == target_label

    area = int(comp_mask.sum())
    radius_fwhm = math.sqrt(max(area, 1) / math.pi)

    # Intensity-weighted 2nd central moments for elongation/eccentricity
    ys, xs = np.nonzero(comp_mask)
    if len(xs) >= 3:
        weights = np.clip(patch[ys, xs] - local_bg, 1e-6, None)
        w_sum = float(weights.sum())
        mx = float((weights * xs).sum() / w_sum)
        my = float((weights * ys).sum() / w_sum)
        dx = xs - mx
        dy = ys - my
        # Add 1/12 pixel-discretization variance to stabilize tiny spots
        mu_xx = float((weights * dx * dx).sum() / w_sum) + (1.0 / 12.0)
        mu_yy = float((weights * dy * dy).sum() / w_sum) + (1.0 / 12.0)
        mu_xy = float((weights * dx * dy).sum() / w_sum)
        trace = mu_xx + mu_yy
        disc = math.sqrt(max(0.0, (mu_xx - mu_yy) ** 2 + 4.0 * mu_xy * mu_xy))
        lam_max = 0.5 * (trace + disc)
        lam_min = max(0.0, 0.5 * (trace - disc))
        ecc = math.sqrt(max(0.0, 1.0 - lam_min / max(lam_max, 1e-12)))
    else:
        ecc = 0.0

    return {
        "Spot_Peak_Intensity": round(spot_peak, 2),
        "Spot_Area_px": area,
        "Spot_Radius_FWHM_px": round(radius_fwhm, 3),
        "Eccentricity": round(float(np.clip(ecc, 0.0, 1.0)), 4),
        "Is_Saturated": is_saturated,
    }


def estimate_drift(
    prev_image: np.ndarray,
    curr_image: np.ndarray,
    upsample_factor: int = 10,
) -> tuple[float, float, float]:
    """Estimate (shift_y, shift_x, error) to align curr_image onto prev_image."""
    shift, error, _ = phase_cross_correlation(
        prev_image,
        curr_image,
        upsample_factor=upsample_factor,
    )
    return float(shift[0]), float(shift[1]), float(error)


def track_blobs(
    prev_blobs: np.ndarray | None,
    curr_blobs: np.ndarray,
    max_distance: float,
    next_id: int = 0,
    drift_yx: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, int]:
    """Match current blobs to previous blobs using drift-compensated 1-to-1 Hungarian assignment.

    `drift_yx` is applied only to the matching distance calculation; returned coordinates
    remain the true unshifted coordinates of `curr_blobs`.
    """
    n_curr = len(curr_blobs)
    if n_curr == 0:
        return np.empty((0, 4), dtype=np.float64), next_id

    if prev_blobs is None or len(prev_blobs) == 0:
        ids = np.arange(next_id, next_id + n_curr, dtype=np.float64)
        return np.column_stack((curr_blobs, ids)), next_id + n_curr

    curr_aligned_yx = curr_blobs[:, :2] + np.array([[drift_yx[0], drift_yx[1]]], dtype=np.float64)
    distances = cdist(prev_blobs[:, :2], curr_aligned_yx)
    sentinel = max_distance * 1e6 + 1e6
    cost_matrix = np.where(distances < max_distance, distances, sentinel)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_ids = np.full(n_curr, -1, dtype=np.int64)

    for prev_idx, curr_idx in zip(row_ind, col_ind):
        if distances[prev_idx, curr_idx] < max_distance:
            assigned_ids[curr_idx] = int(prev_blobs[prev_idx, 3])

    for idx in range(n_curr):
        if assigned_ids[idx] == -1:
            assigned_ids[idx] = next_id
            next_id += 1

    tracked = np.column_stack((curr_blobs, assigned_ids.astype(np.float64)))
    return tracked, next_id


def _draw_tracked_frame(img_bgr: np.ndarray, tracked_blobs: np.ndarray) -> np.ndarray:
    annotated = img_bgr.copy() if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    color = (0, 255, 0)
    for y, x, r, blob_id in tracked_blobs:
        ix, iy, ir = int(round(x)), int(round(y)), max(2, int(round(r)))
        cv2.circle(annotated, (ix, iy), ir, color, 1)
        cv2.putText(
            annotated,
            str(int(blob_id)),
            (ix + 2, max(12, iy - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_detection(
    frames: Sequence[FrameInput],
    output_dir: Path | str,
    det_cfg: DetectionConfig,
    trk_cfg: TrackingConfig,
) -> pd.DataFrame:
    """Run detection-only cleaning, LoG detection, spot morphology, and optional drift tracking."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    tracked_img_dir = out_root / "tracked_images"
    cleaned_img_dir = out_root / "cleaned_detection_images"
    tracking_dir = out_root / "tracking"
    if det_cfg.save_tracked_images:
        tracked_img_dir.mkdir(parents=True, exist_ok=True)
    if det_cfg.save_cleaned_images:
        cleaned_img_dir.mkdir(parents=True, exist_ok=True)
    if trk_cfg.enabled:
        tracking_dir.mkdir(parents=True, exist_ok=True)

    prev_blobs: np.ndarray | None = None
    prev_clean: np.ndarray | None = None
    next_id = 0
    rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []

    for frame_number, frame in enumerate(frames):
        img_bgr = cv2.imread(str(frame.tiff_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise OSError(f"Failed to read TIFF image: {frame.tiff_path}")

        cleaned_0_1, destriped_raw = prepare_detection_image(img_bgr, det_cfg)
        if det_cfg.save_cleaned_images:
            clean_u8 = np.clip(np.rint(cleaned_0_1 * 255.0), 0, 255).astype(np.uint8)
            cv2.imwrite(str(cleaned_img_dir / f"frame_{frame_number:03d}_{frame.stem}_clean.png"), clean_u8)

        raw_proj = project_channels(img_bgr, det_cfg.channel_projection)
        blobs = detect_blobs(cleaned_0_1, det_cfg)

        # Measure morphology on the destriped projection (or raw_proj if cleaning disabled)
        morph_source = destriped_raw if det_cfg.clean_enabled else raw_proj
        morph_list: list[dict[str, float | int | bool]] = []
        keep_indices: list[int] = []
        for idx, (y, x, _r) in enumerate(blobs):
            morph = measure_spot_morphology(morph_source, float(x), float(y))
            if float(morph["Spot_Peak_Intensity"]) >= det_cfg.min_peak_intensity:
                keep_indices.append(idx)
                morph_list.append(morph)

        if len(keep_indices) < len(blobs):
            blobs = blobs[keep_indices]

        # Drift estimation & tracking
        shift_y, shift_x, reg_err = 0.0, 0.0, 0.0
        if trk_cfg.enabled:
            if prev_clean is not None and trk_cfg.register_drift:
                shift_y, shift_x, reg_err = estimate_drift(
                    prev_clean, cleaned_0_1, upsample_factor=trk_cfg.drift_upsample_factor
                )
                logger.info(
                    "Frame %d (%s): estimated stage drift (dy=%+.2f, dx=%+.2f) px.",
                    frame_number,
                    frame.stem,
                    shift_y,
                    shift_x,
                )
            drift_rows.append(
                {
                    "Frame_Number": frame_number,
                    "Filename": frame.stem,
                    "Shift_Y_px": round(shift_y, 3),
                    "Shift_X_px": round(shift_x, 3),
                    "Registration_Error": round(reg_err, 4),
                }
            )
            tracked, next_id = track_blobs(
                prev_blobs,
                blobs,
                max_distance=trk_cfg.max_distance_px,
                next_id=next_id,
                drift_yx=(shift_y, shift_x),
            )
            prev_blobs = tracked
            prev_clean = cleaned_0_1
        else:
            ids = np.arange(len(blobs), dtype=np.float64)
            tracked = (
                np.column_stack((blobs, ids))
                if len(blobs) > 0
                else np.empty((0, 4), dtype=np.float64)
            )

        logger.info(
            "Frame %d (%s): detected %d blob(s) [clean=%s, tracking=%s].",
            frame_number,
            frame.stem,
            len(tracked),
            det_cfg.clean_enabled,
            trk_cfg.enabled,
        )

        for (y, x, r, blob_id), morph in zip(tracked, morph_list):
            rows.append(
                {
                    "Blob_ID": int(blob_id),
                    "Frame_Number": frame_number,
                    "X": float(x),
                    "Y": float(y),
                    "Radius": float(r),
                    **morph,
                    "Filename": frame.stem,
                }
            )

        if det_cfg.save_tracked_images:
            vis_bgr = img_bgr[..., :3] if img_bgr.ndim == 3 else img_bgr
            if vis_bgr.dtype != np.uint8:
                vis_bgr = np.clip(vis_bgr / 257.0, 0, 255).astype(np.uint8)
            annotated = _draw_tracked_frame(vis_bgr, tracked)
            img_out = tracked_img_dir / f"frame_{frame_number:03d}_{frame.stem}_tracked.png"
            cv2.imwrite(str(img_out), annotated)

    if trk_cfg.enabled and drift_rows:
        pd.DataFrame(drift_rows, columns=DRIFT_COLUMNS).to_csv(
            tracking_dir / "frame_drift.csv", index=False
        )

    df = pd.DataFrame(rows, columns=BLOB_COLUMNS)
    csv_path = out_root / "blobs_detected.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved detection table (%d rows) to %s", len(df), csv_path)
    return df
