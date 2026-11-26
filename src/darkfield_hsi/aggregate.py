"""Stage 4b: Physics-driven multi-modal aggregate classification (spectral + morphological)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from darkfield_hsi.config import AggregateConfig

logger = logging.getLogger(__name__)

AGGREGATE_COLUMNS = [
    "Blob_ID",
    "Frame_Number",
    "Filename",
    "Aggregate_Score",
    "Aggregate_Class",
    "Aggregate",
    "Aggregate_Evidence",
]


def _proximity_flags(frame_df: pd.DataFrame, neighbor_factor: float) -> np.ndarray:
    n = len(frame_df)
    if n < 2:
        return np.zeros(n, dtype=bool)
    x = frame_df["X"].to_numpy(dtype=np.float64)
    y = frame_df["Y"].to_numpy(dtype=np.float64)
    r_col = "Spot_Radius_FWHM_px" if "Spot_Radius_FWHM_px" in frame_df.columns else "Radius"
    r = frame_df[r_col].to_numpy(dtype=np.float64)
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    thresh = neighbor_factor * (r[:, None] + r[None, :])
    np.fill_diagonal(dist, np.inf)
    return np.any(dist < thresh, axis=1)


def _is_broadband_core(h_mean: float, s_mean: float, v_mean: float, cfg: AggregateConfig) -> bool:
    is_white = (v_mean >= cfg.v_white_min) and (s_mean <= cfg.s_white_max)
    is_yellow = (
        (cfg.h_yel_lo <= h_mean <= cfg.h_yel_hi)
        and (s_mean >= cfg.s_yel_min)
        and (v_mean >= cfg.v_yel_min)
    )
    return bool(is_white or is_yellow)


def classify_aggregates(
    merged_df: pd.DataFrame,
    output_dir: Path | str,
    cfg: AggregateConfig,
) -> pd.DataFrame:
    """Score and classify nanoparticles using spectral + morphological + optical evidence.

    Gracefully falls back to morphology + foreground-optical scoring when ENVI spectral
    columns are absent (TIFF-only mode). Uses 'Single' (never 'None') for non-aggregates
    so pandas/Excel never coerce labels to NaN.
    """
    pp_dir = Path(output_dir) / "postprocess"
    pp_dir.mkdir(parents=True, exist_ok=True)

    if merged_df.empty:
        empty = pd.DataFrame(columns=AGGREGATE_COLUMNS)
        empty.to_csv(pp_dir / "blobs_aggregate.csv", index=False)
        return empty

    red_shift_nm = cfg.resolved_red_shift_nm()
    broad_fwhm_nm = cfg.resolved_broad_fwhm_nm()

    has_spectra_cols = all(
        c in merged_df.columns for c in ("FWHM_nm", "Max_Wavelength_nm", "Max_Intensity", "Num_Peaks")
    )
    has_morph_cols = all(c in merged_df.columns for c in ("Spot_Area_px", "Eccentricity"))
    has_color_cols = all(c in merged_df.columns for c in ("Hue_mean_deg", "Sat_mean", "Val_mean"))

    records: list[dict[str, object]] = []

    for stem, grp in merged_df.groupby("Filename", sort=False):
        stem_str = str(stem)
        prox = _proximity_flags(grp, cfg.neighbor_factor) if cfg.use_proximity else np.zeros(len(grp), dtype=bool)

        # Frame-relative exposure-invariant baselines
        med_area = (
            float(np.nanmedian(grp["Spot_Area_px"].to_numpy(dtype=np.float64)))
            if has_morph_cols
            else 4.0
        )
        med_area = max(med_area, 1.0)

        med_int = 1.0
        if has_spectra_cols:
            if "Low_SNR" in grp.columns:
                valid_int = grp.loc[~grp["Low_SNR"].astype(bool), "Max_Intensity"].dropna()
            else:
                valid_int = grp["Max_Intensity"].dropna()
            if len(valid_int) > 0:
                med_int = max(float(np.median(valid_int.to_numpy(dtype=np.float64))), 1e-6)

        for local_i, (_, row) in enumerate(grp.iterrows()):
            blob_id = int(row["Blob_ID"])
            frame_num = int(row["Frame_Number"]) if "Frame_Number" in grp.columns else 0

            evidence: list[str] = []
            raw_score = 0.0

            # --- Spectral modality ---
            use_spectral_row = False
            if has_spectra_cols:
                low_snr = bool(row["Low_SNR"]) if "Low_SNR" in grp.columns and pd.notna(row["Low_SNR"]) else False
                fwhm_val = row["FWHM_nm"]
                peak_val = row["Max_Wavelength_nm"]
                int_val = row["Max_Intensity"]
                n_peaks_val = row["Num_Peaks"]
                if (not low_snr) and pd.notna(fwhm_val) and pd.notna(peak_val):
                    use_spectral_row = True
                    if float(fwhm_val) >= broad_fwhm_nm:
                        raw_score += 0.30
                        evidence.append("BroadFWHM")
                    if float(peak_val) >= red_shift_nm:
                        raw_score += 0.25
                        evidence.append("RedShifted")
                    if pd.notna(n_peaks_val) and int(n_peaks_val) >= 2:
                        raw_score += 0.25
                        evidence.append("MultiPeak")
                    if pd.notna(int_val) and float(int_val) >= cfg.rel_intensity_thresh * med_int:
                        raw_score += 0.20
                        evidence.append("HighIntensity")

            # --- Morphological modality ---
            if has_morph_cols:
                area = float(row["Spot_Area_px"])
                ecc = float(row["Eccentricity"])
                if area >= max(cfg.min_large_spot_area_px, cfg.rel_area_thresh * med_area):
                    raw_score += 0.30
                    evidence.append("LargeSpot")
                if area >= cfg.min_eccentricity_area_px and ecc >= cfg.eccentricity_thresh:
                    raw_score += 0.25
                    evidence.append("Elongated")

            # --- Foreground broadband core modality ---
            if has_color_cols:
                h_m = float(row["Hue_mean_deg"])
                s_m = float(row["Sat_mean"])
                v_m = float(row["Val_mean"])
                if _is_broadband_core(h_m, s_m, v_m, cfg):
                    raw_score += 0.20
                    evidence.append("BroadbandCore")

            # --- Optional spatial proximity ---
            if prox[local_i]:
                raw_score += 0.20
                evidence.append("Proximity")

            # Normalize score cleanly in TIFF-only (morphology-only) fallback mode
            if not use_spectral_row:
                morph_denom = 0.75 + (0.20 if cfg.use_proximity else 0.0)
                score = min(1.0, raw_score / morph_denom)
            else:
                score = min(1.0, raw_score)

            if score >= cfg.aggregate_score_thresh:
                agg_class = "Aggregate"
            elif score >= cfg.likely_score_thresh:
                agg_class = "Likely_Aggregate"
            else:
                agg_class = "Single"

            records.append(
                {
                    "Blob_ID": blob_id,
                    "Frame_Number": frame_num,
                    "Filename": stem_str,
                    "Aggregate_Score": round(score, 3),
                    "Aggregate_Class": agg_class,
                    "Aggregate": agg_class == "Aggregate",
                    "Aggregate_Evidence": "+".join(evidence) if evidence else "Single",
                }
            )

    agg_df = pd.DataFrame(records, columns=AGGREGATE_COLUMNS)
    agg_csv = pp_dir / "blobs_aggregate.csv"
    agg_df.to_csv(agg_csv, index=False)

    counts = agg_df["Aggregate_Class"].value_counts().to_dict()
    logger.info(
        "Multi-modal aggregate classification complete (preset=%s, spectral=%s): %s -> %s",
        cfg.material_preset,
        has_spectra_cols,
        counts,
        agg_csv,
    )
    return agg_df
