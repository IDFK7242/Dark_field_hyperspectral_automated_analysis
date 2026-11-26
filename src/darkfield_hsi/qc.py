"""Stage 0: TIFF frame quality-control gate to eliminate noisy/corrupted frames."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from darkfield_hsi.config import QcConfig
from darkfield_hsi.io_utils import FrameInput

logger = logging.getLogger(__name__)

QC_COLUMNS = [
    "Filename",
    "Tiff_Path",
    "Bg_Median",
    "Bg_MAD",
    "HF_Noise_MAD",
    "Stripe_Index",
    "SNR_P995",
    "Laplacian_Focus",
    "Passed_QC",
    "Reject_Reason",
]


@dataclass
class FrameQcResult:
    """Computed frame-level noise, striping, SNR, and focus metrics."""

    Filename: str
    Tiff_Path: str
    Bg_Median: float
    Bg_MAD: float
    HF_Noise_MAD: float
    Stripe_Index: float
    SNR_P995: float
    Laplacian_Focus: float
    Passed_QC: bool = True
    Reject_Reason: str = "Pass"


def _load_projection_u8(tiff_path: Path) -> np.ndarray:
    """Load page 0 of a TIFF and return a 2D float64 max-channel projection in [0, 255]."""
    img = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"Failed to read TIFF for QC: {tiff_path}")

    if img.ndim == 3:
        # Take max across RGB/BGR channels (ignore alpha if 4-channel)
        proj = img[..., :3].max(axis=2)
    else:
        proj = img

    if proj.dtype == np.uint8:
        return proj.astype(np.float64)
    if proj.dtype == np.uint16:
        return proj.astype(np.float64) / 257.0
    max_val = float(np.max(proj)) if proj.size else 1.0
    scale = 255.0 / max_val if max_val > 255.0 else 1.0
    return np.clip(proj.astype(np.float64) * scale, 0.0, 255.0)


def evaluate_frame_qc(frame: FrameInput, cfg: QcConfig) -> FrameQcResult:
    """Compute QC metrics and hard-threshold pass/fail checks for a single TIFF frame."""
    proj = _load_projection_u8(frame.tiff_path)
    proj_u8 = np.clip(np.rint(proj), 0, 255).astype(np.uint8)

    bg_median = float(np.median(proj))
    bg_mad = float(np.median(np.abs(proj - bg_median)))

    blur3 = cv2.medianBlur(proj_u8, 3).astype(np.float64)
    hf_noise_mad = float(np.median(np.abs(proj - blur3)))

    col_medians = np.median(proj, axis=0)
    stripe_index = float(np.std(col_medians) / max(bg_mad, 1.0))

    p995 = float(np.percentile(proj, 99.5))
    snr_p995 = float((p995 - bg_median) / max(bg_mad, 1.0))

    laplacian_focus = float(cv2.Laplacian(proj, cv2.CV_64F).var())

    reasons: list[str] = []
    if bg_median > cfg.max_bg_median:
        reasons.append(f"HighBgMedian({bg_median:.1f}>{cfg.max_bg_median:.1f})")
    if bg_mad > cfg.max_noise_mad:
        reasons.append(f"HighBgNoise({bg_mad:.1f}>{cfg.max_noise_mad:.1f})")
    if hf_noise_mad > cfg.max_hf_noise_mad:
        reasons.append(f"HighHFNoise({hf_noise_mad:.1f}>{cfg.max_hf_noise_mad:.1f})")
    if stripe_index > cfg.max_stripe_index:
        reasons.append(f"SevereStriping({stripe_index:.2f}>{cfg.max_stripe_index:.2f})")
    if snr_p995 < cfg.min_frame_snr:
        reasons.append(f"LowSNR({snr_p995:.1f}<{cfg.min_frame_snr:.1f})")
    if laplacian_focus < cfg.min_laplacian_var:
        reasons.append(f"LowFocus({laplacian_focus:.1f}<{cfg.min_laplacian_var:.1f})")

    passed = len(reasons) == 0
    return FrameQcResult(
        Filename=frame.stem,
        Tiff_Path=str(frame.tiff_path),
        Bg_Median=round(bg_median, 3),
        Bg_MAD=round(bg_mad, 3),
        HF_Noise_MAD=round(hf_noise_mad, 3),
        Stripe_Index=round(stripe_index, 3),
        SNR_P995=round(snr_p995, 3),
        Laplacian_Focus=round(laplacian_focus, 3),
        Passed_QC=passed,
        Reject_Reason="Pass" if passed else ";".join(reasons),
    )


def _apply_relative_outlier_checks(results: list[FrameQcResult], cfg: QcConfig) -> None:
    """Flag dataset-relative MAD outliers when >= 3 frames are evaluated."""
    if not cfg.use_relative_outliers or len(results) < 3:
        return

    high_bad_fields = [
        ("Bg_Median", "OutlierBgMedian"),
        ("Bg_MAD", "OutlierBgNoise"),
        ("HF_Noise_MAD", "OutlierHFNoise"),
        ("Stripe_Index", "OutlierStriping"),
    ]
    for field_name, tag in high_bad_fields:
        vals = np.array([getattr(r, field_name) for r in results], dtype=np.float64)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        # Use a minimum MAD floor to avoid flagging tiny benign numerical differences
        scale = max(mad, 0.5)
        for r, v in zip(results, vals):
            mod_z = 0.6745 * (v - med) / scale
            if mod_z > cfg.outlier_z_thresh:
                r.Passed_QC = False
                reason = f"{tag}(z={mod_z:.1f})"
                r.Reject_Reason = reason if r.Reject_Reason == "Pass" else f"{r.Reject_Reason};{reason}"

    # Low SNR outlier check
    snr_vals = np.array([r.SNR_P995 for r in results], dtype=np.float64)
    snr_med = float(np.median(snr_vals))
    snr_mad = float(np.median(np.abs(snr_vals - snr_med)))
    snr_scale = max(snr_mad, 0.5)
    for r, v in zip(results, snr_vals):
        mod_z = 0.6745 * (v - snr_med) / snr_scale
        if mod_z < -cfg.outlier_z_thresh:
            r.Passed_QC = False
            reason = f"OutlierLowSNR(z={mod_z:.1f})"
            r.Reject_Reason = reason if r.Reject_Reason == "Pass" else f"{r.Reject_Reason};{reason}"


def run_frame_qc(
    frames: Sequence[FrameInput],
    output_dir: Path | str,
    cfg: QcConfig,
) -> tuple[list[FrameInput], pd.DataFrame]:
    """Evaluate all discovered TIFF frames, write qc/frame_qc.csv, and return accepted frames."""
    qc_dir = Path(output_dir) / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    results = [evaluate_frame_qc(frame, cfg) for frame in frames]
    _apply_relative_outlier_checks(results, cfg)

    qc_df = pd.DataFrame([asdict(r) for r in results], columns=QC_COLUMNS)
    qc_csv = qc_dir / "frame_qc.csv"
    qc_df.to_csv(qc_csv, index=False)

    if not cfg.enabled:
        logger.info("Frame QC disabled; passing all %d frame(s) (metrics logged to %s).", len(frames), qc_csv)
        return list(frames), qc_df

    accepted: list[FrameInput] = []
    rejected_count = 0
    for frame, res in zip(frames, results):
        if res.Passed_QC:
            accepted.append(frame)
        else:
            rejected_count += 1
            logger.warning("Frame QC REJECTED %s: %s", frame.stem, res.Reject_Reason)

    logger.info(
        "Frame QC complete: %d/%d frame(s) accepted (%d rejected) -> %s",
        len(accepted),
        len(frames),
        rejected_count,
        qc_csv,
    )
    return accepted, qc_df
