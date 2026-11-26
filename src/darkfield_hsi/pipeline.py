"""End-to-end single-experiment and multi-experiment batch pipeline orchestrators."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from darkfield_hsi.aggregate import classify_aggregates
from darkfield_hsi.config import PipelineConfig, load_config, save_config
from darkfield_hsi.detection import run_detection
from darkfield_hsi.envi import run_envi_extraction
from darkfield_hsi.io_utils import TIFF_SUFFIXES, discover_inputs
from darkfield_hsi.postprocess import run_postprocess
from darkfield_hsi.qc import run_frame_qc
from darkfield_hsi.reporting import (
    compute_tracked_shifts,
    generate_html_report,
    generate_summary_plots,
)
from darkfield_hsi.spectra import run_spectral_analysis

logger = logging.getLogger(__name__)

MERGE_KEYS = ["Filename", "Frame_Number", "Blob_ID"]


def run_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Execute the full v0.2 darkfield-hsi pipeline (Stages 0–5) and write blobs_summary.csv."""
    resolved_cfg = cfg if cfg is not None else load_config()
    resolved_cfg.validate()

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    save_config(resolved_cfg, out_root / "run_config.yaml")

    all_frames = discover_inputs(input_dir, file_order=resolved_cfg.input.file_order)

    # Stage 0: Frame QC gate
    logger.info("=== Stage 0: TIFF Frame Quality Control ===")
    frames, qc_df = run_frame_qc(all_frames, out_root, resolved_cfg.qc)
    if not frames:
        raise RuntimeError(
            f"All {len(all_frames)} TIFF frame(s) were rejected by Frame QC. "
            f"Inspect {out_root / 'qc' / 'frame_qc.csv'}, relax qc thresholds, or pass --no-qc."
        )

    # Stage 1: Detection-only cleaning, LoG detection, spot morphology, drift registration & tracking
    logger.info("=== Stage 1: Detection, Spot Morphology & Tracking ===")
    blobs_df = run_detection(frames, out_root, resolved_cfg.detection, resolved_cfg.tracking)

    # Stages 2 & 3: ENVI Z-profiles and spectral analysis (when ENVI cubes match)
    spec_df: pd.DataFrame | None = None
    if any(f.has_envi for f in frames) and not blobs_df.empty:
        logger.info("=== Stage 2: ENVI Z-Profile Extraction ===")
        z_csvs = run_envi_extraction(frames, blobs_df, out_root, resolved_cfg.envi)
        if z_csvs:
            logger.info("=== Stage 3: Spectral Smoothing, SNR, Multi-Peak & Per-Blob Exports ===")
            spec_df = run_spectral_analysis(
                out_root / "z_profiles", out_root, resolved_cfg.spectra
            )
    else:
        logger.info("No matched ENVI hyperspectral cubes (or no blobs); skipping Stages 2 and 3.")

    # Stage 4a: Foreground-masked color classification & centered crops
    logger.info("=== Stage 4a: Foreground Color & Centered Crops ===")
    pp_df = run_postprocess(frames, blobs_df, out_root, resolved_cfg.postprocess)

    # Assemble combined feature table for Stage 4b multi-modal aggregate scoring
    summary = blobs_df.copy()
    if not pp_df.empty:
        summary = summary.merge(pp_df, on=MERGE_KEYS, how="left", validate="one_to_one")
    if spec_df is not None and not spec_df.empty:
        summary = summary.merge(spec_df, on=MERGE_KEYS, how="left", validate="one_to_one")

    # Stage 4b: Physics-driven multi-modal aggregate classification
    logger.info("=== Stage 4b: Physics-Driven Multi-Modal Aggregate Classification ===")
    agg_df = classify_aggregates(summary, out_root, resolved_cfg.aggregate)
    if not agg_df.empty:
        summary = summary.merge(agg_df, on=MERGE_KEYS, how="left", validate="one_to_one")

    summary_path = out_root / "blobs_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Stage 5: Time-series tracked shifts, population summary figures, and interactive HTML report
    logger.info("=== Stage 5: Tracked Shifts & Reporting ===")
    shifts_df = compute_tracked_shifts(summary, out_root, resolved_cfg.tracking)

    if resolved_cfg.reporting.enabled:
        summary_plots: list[Path] = []
        if resolved_cfg.reporting.save_summary_plots:
            summary_plots = generate_summary_plots(summary, shifts_df, out_root)
        if resolved_cfg.reporting.save_html_report:
            generate_html_report(
                summary,
                qc_df,
                shifts_df,
                out_root,
                summary_plots,
                resolved_cfg.reporting,
                frames=frames,
            )

    logger.info("Pipeline complete! Merged summary (%d rows) -> %s", len(summary), summary_path)
    return summary


def _dir_has_tiffs(directory: Path) -> bool:
    for candidate in (directory / "tiff", directory):
        if candidate.is_dir():
            if any(p.is_file() and p.suffix.lower() in TIFF_SUFFIXES for p in candidate.iterdir()):
                return True
    return False


def discover_batch_samples(parent_dir: Path | str) -> list[Path]:
    """Find immediate subdirectories of parent_dir that contain TIFF experiment data."""
    root = Path(parent_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Batch parent directory does not exist: {root}")
    samples = sorted(
        (d for d in root.iterdir() if d.is_dir() and _dir_has_tiffs(d)),
        key=lambda p: p.name,
    )
    if not samples:
        raise FileNotFoundError(
            f"No sample subdirectories containing TIFFs found inside {root}"
        )
    return samples


def run_batch(
    parent_dir: Path | str,
    output_root: Path | str,
    cfg: PipelineConfig | None = None,
) -> pd.DataFrame:
    """Run the full pipeline across multiple sample subdirectories and emit batch_summary.csv."""
    resolved_cfg = cfg if cfg is not None else load_config()
    resolved_cfg.validate()

    out_base = Path(output_root)
    out_base.mkdir(parents=True, exist_ok=True)
    save_config(resolved_cfg, out_base / "batch_config.yaml")

    sample_dirs = discover_batch_samples(parent_dir)
    logger.info("Batch mode: discovered %d sample(s) in %s", len(sample_dirs), parent_dir)

    all_summaries: list[pd.DataFrame] = []
    overview_rows: list[dict[str, object]] = []

    for idx, sample_dir in enumerate(sample_dirs, start=1):
        sample_name = sample_dir.name
        logger.info(">>> [%d/%d] Processing sample: %s", idx, len(sample_dirs), sample_name)
        sample_out = out_base / sample_name
        df = run_pipeline(sample_dir, sample_out, resolved_cfg)
        tagged = df.copy()
        tagged.insert(0, "Sample", sample_name)
        all_summaries.append(tagged)

        n_blobs = len(df)
        n_agg = int((df["Aggregate_Class"] == "Aggregate").sum()) if "Aggregate_Class" in df.columns else 0
        n_likely = int((df["Aggregate_Class"] == "Likely_Aggregate").sum()) if "Aggregate_Class" in df.columns else 0
        med_peak = (
            round(float(df["Max_Wavelength_nm"].median()), 2)
            if ("Max_Wavelength_nm" in df.columns and df["Max_Wavelength_nm"].notna().any())
            else None
        )
        med_fwhm = (
            round(float(df["FWHM_nm"].median()), 2)
            if ("FWHM_nm" in df.columns and df["FWHM_nm"].notna().any())
            else None
        )
        overview_rows.append(
            {
                "Sample": sample_name,
                "Total_Blobs": n_blobs,
                "Single_Count": n_blobs - n_agg - n_likely,
                "Likely_Aggregate_Count": n_likely,
                "Aggregate_Count": n_agg,
                "Aggregate_Fraction": round(n_agg / n_blobs, 4) if n_blobs else 0.0,
                "Median_Peak_nm": med_peak,
                "Median_FWHM_nm": med_fwhm,
            }
        )

    batch_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    batch_csv = out_base / "batch_summary.csv"
    overview_csv = out_base / "batch_overview.csv"
    batch_df.to_csv(batch_csv, index=False)
    pd.DataFrame(overview_rows).to_csv(overview_csv, index=False)
    logger.info(
        "Batch complete! %d sample(s), %d total blobs -> %s",
        len(sample_dirs),
        len(batch_df),
        batch_csv,
    )
    return batch_df
