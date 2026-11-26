"""Command-line interface for darkfield-hsi."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from darkfield_hsi.aggregate import classify_aggregates
from darkfield_hsi.config import PipelineConfig, load_config, save_config
from darkfield_hsi.detection import run_detection
from darkfield_hsi.envi import run_envi_extraction
from darkfield_hsi.io_utils import discover_inputs
from darkfield_hsi.pipeline import MERGE_KEYS, run_batch, run_pipeline
from darkfield_hsi.postprocess import run_postprocess
from darkfield_hsi.qc import run_frame_qc
from darkfield_hsi.reporting import regenerate_report
from darkfield_hsi.spectra import run_spectral_analysis


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_workers(val: str) -> int | str:
    if val.strip().lower() == "auto":
        return "auto"
    return int(val)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file (uses defaults if omitted).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )


def _add_run_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-qc", action="store_true", help="Disable Stage 0 TIFF frame QC rejection.")
    parser.add_argument("--no-track", action="store_true", help="Disable cross-frame blob tracking.")
    parser.add_argument("--no-plots", action="store_true", help="Disable per-blob spectrum PNG plots.")
    parser.add_argument("--no-per-blob-csv", action="store_true", help="Disable per-blob spectrum CSVs.")
    parser.add_argument("--no-report", action="store_true", help="Disable Stage 5 summary plots & HTML report.")
    parser.add_argument(
        "--workers",
        type=_parse_workers,
        default=None,
        help="Override parallel worker count ('auto' or integer >= 1).",
    )


def _apply_run_overrides(cfg: PipelineConfig, args: argparse.Namespace) -> None:
    if args.no_qc:
        cfg.qc.enabled = False
    if args.no_track:
        cfg.tracking.enabled = False
    if args.no_plots:
        cfg.spectra.save_plots = False
    if args.no_per_blob_csv:
        cfg.spectra.save_per_blob_csv = False
    if args.no_report:
        cfg.reporting.enabled = False
    if args.workers is not None:
        cfg.spectra.workers = args.workers
    cfg.validate()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darkfield-hsi",
        description="Automated dark-field hyperspectral imaging (HSI) nanoparticle processing CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init-config
    p_init = sub.add_parser("init-config", help="Write a default YAML configuration file.")
    p_init.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("config.yaml"),
        help="Output path for the generated YAML config (default: config.yaml).",
    )
    p_init.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

    # qc (Stage 0 dry-run / inspection)
    p_qc = sub.add_parser("qc", help="Run Stage 0 TIFF frame quality-control inspection only.")
    p_qc.add_argument("-i", "--input-dir", type=Path, required=True, help="Input data directory.")
    p_qc.add_argument("-o", "--output-dir", type=Path, required=True, help="Output directory for qc/frame_qc.csv.")
    _add_common_args(p_qc)

    # run (full single-experiment pipeline)
    p_run = sub.add_parser("run", help="Run the full end-to-end pipeline (Stages 0–5).")
    p_run.add_argument("-i", "--input-dir", type=Path, required=True, help="Input data directory.")
    p_run.add_argument("-o", "--output-dir", type=Path, required=True, help="Output results directory.")
    _add_run_override_args(p_run)
    _add_common_args(p_run)

    # batch (multi-experiment folder runner)
    p_batch = sub.add_parser("batch", help="Run the full pipeline across multiple sample subdirectories.")
    p_batch.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Parent directory containing one subdirectory per sample/experiment.",
    )
    p_batch.add_argument("-o", "--output-dir", type=Path, required=True, help="Output root directory.")
    _add_run_override_args(p_batch)
    _add_common_args(p_batch)

    # detect (Stage 1)
    p_det = sub.add_parser("detect", help="Run Stage 0 QC + Stage 1 detection & tracking.")
    p_det.add_argument("-i", "--input-dir", type=Path, required=True, help="Input data directory.")
    p_det.add_argument("-o", "--output-dir", type=Path, required=True, help="Output results directory.")
    p_det.add_argument("--no-qc", action="store_true", help="Disable Stage 0 TIFF frame QC.")
    p_det.add_argument("--no-track", action="store_true", help="Disable cross-frame blob tracking.")
    _add_common_args(p_det)

    # extract-envi (Stage 2)
    p_envi = sub.add_parser("extract-envi", help="Run Stage 2: ENVI Z-profile extraction.")
    p_envi.add_argument("-i", "--input-dir", type=Path, required=True, help="Input data directory.")
    p_envi.add_argument("--blobs", type=Path, required=True, help="Path to blobs_detected.csv.")
    p_envi.add_argument("-o", "--output-dir", type=Path, required=True, help="Output results directory.")
    p_envi.add_argument("--save-plots", action="store_true", help="Save raw Z-profile plots.")
    _add_common_args(p_envi)

    # analyze-spectra (Stage 3)
    p_spec = sub.add_parser("analyze-spectra", help="Run Stage 3: spectral analysis, per-blob CSVs & plots.")
    p_spec.add_argument(
        "--z-profiles",
        type=Path,
        required=True,
        help="Directory containing *_z_profile.csv files.",
    )
    p_spec.add_argument("-o", "--output-dir", type=Path, required=True, help="Output results directory.")
    p_spec.add_argument("--no-plots", action="store_true", help="Disable per-blob spectrum PNG plots.")
    p_spec.add_argument("--no-per-blob-csv", action="store_true", help="Disable per-blob spectrum CSVs.")
    p_spec.add_argument(
        "--workers",
        type=_parse_workers,
        default=None,
        help="Override parallel worker count ('auto' or integer >= 1).",
    )
    _add_common_args(p_spec)

    # postprocess (Stage 4a + 4b)
    p_pp = sub.add_parser(
        "postprocess",
        help="Run Stage 4: foreground color/crops and physics-driven multi-modal aggregate classification.",
    )
    p_pp.add_argument("-i", "--input-dir", type=Path, required=True, help="Input data directory.")
    p_pp.add_argument("--blobs", type=Path, required=True, help="Path to blobs_detected.csv.")
    p_pp.add_argument(
        "--spectra",
        type=Path,
        default=None,
        help="Optional path to spectra/spectral_metrics.csv for multi-modal aggregate scoring.",
    )
    p_pp.add_argument("-o", "--output-dir", type=Path, required=True, help="Output results directory.")
    _add_common_args(p_pp)

    # report (Stage 5 fast regeneration)
    p_rep = sub.add_parser(
        "report",
        help="Regenerate summary plots, FOV PNGs, and interactive reports/report.html on an existing run.",
    )
    p_rep.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Existing run directory containing blobs_summary.csv.",
    )
    p_rep.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=None,
        help="Optional original input directory to export/refresh raw dark-field FOV PNGs.",
    )
    _add_common_args(p_rep)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "init-config":
        out = save_config(PipelineConfig(), args.output)
        logging.getLogger(__name__).info("Wrote default configuration to %s", out)
        return 0

    cfg = load_config(args.config)

    if args.command == "qc":
        frames = discover_inputs(args.input_dir, file_order=cfg.input.file_order)
        run_frame_qc(frames, args.output_dir, cfg.qc)
        return 0

    if args.command == "run":
        _apply_run_overrides(cfg, args)
        run_pipeline(args.input_dir, args.output_dir, cfg)
        return 0

    if args.command == "batch":
        _apply_run_overrides(cfg, args)
        run_batch(args.input_dir, args.output_dir, cfg)
        return 0

    if args.command == "detect":
        if args.no_qc:
            cfg.qc.enabled = False
        if args.no_track:
            cfg.tracking.enabled = False
        frames = discover_inputs(args.input_dir, file_order=cfg.input.file_order)
        accepted, _ = run_frame_qc(frames, args.output_dir, cfg.qc)
        if not accepted:
            raise RuntimeError("All TIFF frames failed QC; inspect qc/frame_qc.csv or pass --no-qc.")
        run_detection(accepted, args.output_dir, cfg.detection, cfg.tracking)
        return 0

    if args.command == "extract-envi":
        if args.save_plots:
            cfg.envi.save_plots = True
        frames = discover_inputs(args.input_dir, file_order=cfg.input.file_order)
        blobs_df = pd.read_csv(args.blobs)
        run_envi_extraction(frames, blobs_df, args.output_dir, cfg.envi)
        return 0

    if args.command == "analyze-spectra":
        if args.no_plots:
            cfg.spectra.save_plots = False
        if args.no_per_blob_csv:
            cfg.spectra.save_per_blob_csv = False
        if args.workers is not None:
            cfg.spectra.workers = args.workers
        cfg.validate()
        run_spectral_analysis(args.z_profiles, args.output_dir, cfg.spectra)
        return 0

    if args.command == "postprocess":
        frames = discover_inputs(args.input_dir, file_order=cfg.input.file_order)
        blobs_df = pd.read_csv(args.blobs)
        pp_df = run_postprocess(frames, blobs_df, args.output_dir, cfg.postprocess)
        merged = blobs_df.merge(pp_df, on=MERGE_KEYS, how="left", validate="one_to_one")
        if args.spectra is not None and args.spectra.is_file():
            spec_df = pd.read_csv(args.spectra)
            merged = merged.merge(spec_df, on=MERGE_KEYS, how="left", validate="one_to_one")
        classify_aggregates(merged, args.output_dir, cfg.aggregate)
        return 0

    if args.command == "report":
        regenerate_report(args.output_dir, input_dir=args.input_dir, cfg=cfg)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
