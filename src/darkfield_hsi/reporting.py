"""Stage 5: Time-series tracked shifts, population summary figures, and report regeneration."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from darkfield_hsi.config import PipelineConfig, ReportingConfig, TrackingConfig
from darkfield_hsi.html_report import build_html_report
from darkfield_hsi.io_utils import FrameInput, discover_inputs

logger = logging.getLogger(__name__)

SHIFT_COLUMNS = [
    "Blob_ID",
    "Frames_Seen",
    "First_Frame",
    "Last_Frame",
    "First_Filename",
    "Last_Filename",
    "X_start",
    "Y_start",
    "X_end",
    "Y_end",
    "Displacement_px",
    "Color_start",
    "Color_end",
    "Aggregate_Class_start",
    "Aggregate_Class_end",
    "Aggregate_Score_start",
    "Aggregate_Score_end",
    "Peak_nm_start",
    "Peak_nm_end",
    "Delta_Peak_nm",
    "FWHM_nm_start",
    "FWHM_nm_end",
    "Delta_FWHM_nm",
    "Intensity_start",
    "Intensity_end",
    "Delta_Intensity",
]


def _save_tracked_overlay_plot(
    blob_id: int,
    frame_rows: pd.DataFrame,
    out_root: Path,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for _, row in frame_rows.iterrows():
        rel_csv = str(row.get("Spectrum_CSV", "") or "")
        if not rel_csv:
            continue
        spec_path = out_root / rel_csv
        if not spec_path.is_file():
            continue
        sdf = pd.read_csv(spec_path)
        x_col = "Wavelength_nm" if "Wavelength_nm" in sdf.columns else "Band_Index"
        lbl = f"F{int(row['Frame_Number'])}: {row['Filename']}"
        if "Max_Wavelength_nm" in row and pd.notna(row["Max_Wavelength_nm"]):
            lbl += f" (peak {float(row['Max_Wavelength_nm']):.1f} nm)"
        ax.plot(sdf[x_col], sdf["Smoothed_Intensity"], linewidth=1.6, label=lbl)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_title(f"Tracked Time-Series Spectra — Blob ID {blob_id}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Smoothed Intensity")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def compute_tracked_shifts(
    summary_df: pd.DataFrame,
    output_dir: Path | str,
    trk_cfg: TrackingConfig,
) -> pd.DataFrame:
    """Compute per-Blob_ID kinetic shifts across >= 2 tracked frames."""
    out_root = Path(output_dir)
    tracking_dir = out_root / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    if not trk_cfg.enabled or summary_df.empty or summary_df["Frame_Number"].nunique() < 2:
        empty = pd.DataFrame(columns=SHIFT_COLUMNS)
        empty.to_csv(tracking_dir / "tracked_shifts.csv", index=False)
        return empty

    has_spectra = "Max_Wavelength_nm" in summary_df.columns
    rows: list[dict[str, object]] = []
    grouped_for_overlay: list[tuple[int, float, pd.DataFrame]] = []

    for blob_id, grp in summary_df.groupby("Blob_ID", sort=True):
        if len(grp) < 2:
            continue
        g = grp.sort_values("Frame_Number")
        first = g.iloc[0]
        last = g.iloc[-1]

        disp = math.hypot(float(last["X"]) - float(first["X"]), float(last["Y"]) - float(first["Y"]))

        peak_s = float(first["Max_Wavelength_nm"]) if (has_spectra and pd.notna(first["Max_Wavelength_nm"])) else np.nan
        peak_e = float(last["Max_Wavelength_nm"]) if (has_spectra and pd.notna(last["Max_Wavelength_nm"])) else np.nan
        d_peak = (peak_e - peak_s) if (not math.isnan(peak_s) and not math.isnan(peak_e)) else np.nan

        fwhm_s = float(first["FWHM_nm"]) if (has_spectra and pd.notna(first["FWHM_nm"])) else np.nan
        fwhm_e = float(last["FWHM_nm"]) if (has_spectra and pd.notna(last["FWHM_nm"])) else np.nan
        d_fwhm = (fwhm_e - fwhm_s) if (not math.isnan(fwhm_s) and not math.isnan(fwhm_e)) else np.nan

        int_s = float(first["Max_Intensity"]) if (has_spectra and pd.notna(first["Max_Intensity"])) else np.nan
        int_e = float(last["Max_Intensity"]) if (has_spectra and pd.notna(last["Max_Intensity"])) else np.nan
        d_int = (int_e - int_s) if (not math.isnan(int_s) and not math.isnan(int_e)) else np.nan

        rows.append(
            {
                "Blob_ID": int(blob_id),
                "Frames_Seen": int(len(g)),
                "First_Frame": int(first["Frame_Number"]),
                "Last_Frame": int(last["Frame_Number"]),
                "First_Filename": str(first["Filename"]),
                "Last_Filename": str(last["Filename"]),
                "X_start": float(first["X"]),
                "Y_start": float(first["Y"]),
                "X_end": float(last["X"]),
                "Y_end": float(last["Y"]),
                "Displacement_px": round(disp, 3),
                "Color_start": str(first.get("Color", "")),
                "Color_end": str(last.get("Color", "")),
                "Aggregate_Class_start": str(first.get("Aggregate_Class", "")),
                "Aggregate_Class_end": str(last.get("Aggregate_Class", "")),
                "Aggregate_Score_start": float(first.get("Aggregate_Score", np.nan)),
                "Aggregate_Score_end": float(last.get("Aggregate_Score", np.nan)),
                "Peak_nm_start": round(peak_s, 3) if not math.isnan(peak_s) else np.nan,
                "Peak_nm_end": round(peak_e, 3) if not math.isnan(peak_e) else np.nan,
                "Delta_Peak_nm": round(d_peak, 3) if not math.isnan(d_peak) else np.nan,
                "FWHM_nm_start": round(fwhm_s, 3) if not math.isnan(fwhm_s) else np.nan,
                "FWHM_nm_end": round(fwhm_e, 3) if not math.isnan(fwhm_e) else np.nan,
                "Delta_FWHM_nm": round(d_fwhm, 3) if not math.isnan(d_fwhm) else np.nan,
                "Intensity_start": round(int_s, 3) if not math.isnan(int_s) else np.nan,
                "Intensity_end": round(int_e, 3) if not math.isnan(int_e) else np.nan,
                "Delta_Intensity": round(d_int, 3) if not math.isnan(d_int) else np.nan,
            }
        )

        if trk_cfg.save_overlay_plots and has_spectra:
            sort_key = abs(d_peak) if not math.isnan(d_peak) else 0.0
            grouped_for_overlay.append((int(blob_id), sort_key, g))

    shifts_df = pd.DataFrame(rows, columns=SHIFT_COLUMNS)
    shifts_csv = tracking_dir / "tracked_shifts.csv"
    shifts_df.to_csv(shifts_csv, index=False)
    logger.info("Computed kinetic shifts for %d tracked blob(s) -> %s", len(shifts_df), shifts_csv)

    if trk_cfg.save_overlay_plots and grouped_for_overlay and trk_cfg.max_overlay_plots > 0:
        overlay_dir = tracking_dir / "overlay_plots"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        grouped_for_overlay.sort(key=lambda item: item[1], reverse=True)
        for blob_id, _, g in grouped_for_overlay[: trk_cfg.max_overlay_plots]:
            _save_tracked_overlay_plot(
                blob_id, g, out_root, overlay_dir / f"tracked_b{blob_id:05d}_overlay.png"
            )
        logger.info(
            "Saved %d tracked overlay plot(s) -> %s",
            min(len(grouped_for_overlay), trk_cfg.max_overlay_plots),
            overlay_dir,
        )

    return shifts_df


def generate_summary_plots(
    summary_df: pd.DataFrame,
    shifts_df: pd.DataFrame,
    output_dir: Path | str,
) -> list[Path]:
    """Generate publication-ready population summary PNG charts under reports/summary_plots/."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = Path(output_dir) / "reports" / "summary_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if summary_df.empty:
        return written

    # 1. Color & Aggregate Class breakdown per frame
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if "Aggregate_Class" in summary_df.columns:
        agg_ct = pd.crosstab(summary_df["Filename"], summary_df["Aggregate_Class"])
        col_order = [c for c in ("Single", "Likely_Aggregate", "Aggregate") if c in agg_ct.columns]
        agg_ct = agg_ct[col_order]
        agg_ct.plot(
            kind="bar",
            ax=axes[0],
            color={"Single": "#2e8b57", "Likely_Aggregate": "#e69f00", "Aggregate": "#d62728"},
        )
        axes[0].set_title("Aggregate Classification by Frame")
        axes[0].set_ylabel("Blob Count")
        axes[0].tick_params(axis="x", rotation=20)
        axes[0].grid(axis="y", alpha=0.35)

    if "Color" in summary_df.columns:
        top_colors = summary_df["Color"].value_counts().head(8).index.tolist()
        col_ct = pd.crosstab(
            summary_df["Filename"],
            summary_df["Color"].where(summary_df["Color"].isin(top_colors), "Other"),
        )
        col_ct.plot(kind="bar", ax=axes[1])
        axes[1].set_title("Foreground Color Distribution by Frame")
        axes[1].set_ylabel("Blob Count")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].grid(axis="y", alpha=0.35)

    fig.tight_layout()
    p_break = plots_dir / "color_and_aggregate_breakdown.png"
    fig.savefig(p_break, dpi=130)
    plt.close(fig)
    written.append(p_break)

    # 2. Peak wavelength distribution per frame
    if "Max_Wavelength_nm" in summary_df.columns and summary_df["Max_Wavelength_nm"].notna().any():
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for fname, grp in summary_df.groupby("Filename", sort=False):
            vals = grp["Max_Wavelength_nm"].dropna()
            if len(vals) == 0:
                continue
            med = float(vals.median())
            ax.hist(vals, bins=40, alpha=0.5, label=f"{fname} (med={med:.1f} nm)")
            ax.axvline(med, linestyle="--", linewidth=1.2)
        ax.set_title("LSPR Peak Wavelength Distribution")
        ax.set_xlabel("Peak Wavelength (nm)")
        ax.set_ylabel("Blob Count")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8.5)
        fig.tight_layout()
        p_wl = plots_dir / "peak_wavelength_distribution.png"
        fig.savefig(p_wl, dpi=130)
        plt.close(fig)
        written.append(p_wl)

    # 3. FWHM vs Peak Wavelength scatter colored by Aggregate_Class
    if (
        "Max_Wavelength_nm" in summary_df.columns
        and "FWHM_nm" in summary_df.columns
        and summary_df["FWHM_nm"].notna().any()
    ):
        fig, ax = plt.subplots(figsize=(9, 5.2))
        palette = {"Single": "#2e8b57", "Likely_Aggregate": "#e69f00", "Aggregate": "#d62728"}
        cls_col = "Aggregate_Class" if "Aggregate_Class" in summary_df.columns else None
        if cls_col:
            for cls_name in ("Single", "Likely_Aggregate", "Aggregate"):
                sub = summary_df[summary_df[cls_col] == cls_name]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["Max_Wavelength_nm"],
                    sub["FWHM_nm"],
                    s=16,
                    alpha=0.55,
                    color=palette[cls_name],
                    label=f"{cls_name} (n={len(sub)})",
                )
        else:
            ax.scatter(summary_df["Max_Wavelength_nm"], summary_df["FWHM_nm"], s=16, alpha=0.5)
        ax.set_title("FWHM vs. Peak Wavelength")
        ax.set_xlabel("Peak Wavelength (nm)")
        ax.set_ylabel("FWHM (nm)")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=9)
        fig.tight_layout()
        p_scat = plots_dir / "fwhm_vs_peak_scatter.png"
        fig.savefig(p_scat, dpi=130)
        plt.close(fig)
        written.append(p_scat)

    # 4. Tracked shifts distribution
    if (
        not shifts_df.empty
        and "Delta_Peak_nm" in shifts_df.columns
        and shifts_df["Delta_Peak_nm"].notna().any()
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        dp = shifts_df["Delta_Peak_nm"].dropna()
        axes[0].hist(dp, bins=40, color="#1f77b4", alpha=0.75)
        axes[0].axvline(0.0, color="black", linestyle=":", linewidth=1.0)
        axes[0].axvline(float(dp.median()), color="crimson", linestyle="--", label=f"Median={dp.median():+.1f} nm")
        axes[0].set_title(f"Tracked Peak Shift Δλ (n={len(dp)})")
        axes[0].set_xlabel("Δ Peak Wavelength (nm)")
        axes[0].set_ylabel("Tracked Blob Count")
        axes[0].grid(True, alpha=0.35)
        axes[0].legend()

        dfw = shifts_df["Delta_FWHM_nm"].dropna()
        if len(dfw) > 0:
            axes[1].hist(dfw, bins=40, color="#9467bd", alpha=0.75)
            axes[1].axvline(0.0, color="black", linestyle=":", linewidth=1.0)
            axes[1].axvline(float(dfw.median()), color="crimson", linestyle="--", label=f"Median={dfw.median():+.1f} nm")
            axes[1].set_title(f"Tracked Broadening ΔFWHM (n={len(dfw)})")
            axes[1].set_xlabel("Δ FWHM (nm)")
            axes[1].set_ylabel("Tracked Blob Count")
            axes[1].grid(True, alpha=0.35)
            axes[1].legend()

        fig.tight_layout()
        p_sh = plots_dir / "tracked_shifts_distribution.png"
        fig.savefig(p_sh, dpi=130)
        plt.close(fig)
        written.append(p_sh)

    logger.info("Generated %d summary plot(s) -> %s", len(written), plots_dir)
    return written


def generate_html_report(
    summary_df: pd.DataFrame,
    qc_df: pd.DataFrame | None,
    shifts_df: pd.DataFrame | None,
    output_dir: Path | str,
    summary_plots: list[Path],
    rep_cfg: ReportingConfig,
    frames: Sequence[FrameInput] | None = None,
) -> Path:
    """Delegate standalone interactive HTML workstation generation to html_report."""
    return build_html_report(
        summary_df=summary_df,
        qc_df=qc_df,
        shifts_df=shifts_df,
        output_dir=output_dir,
        summary_plots=summary_plots,
        rep_cfg=rep_cfg,
        frames=frames,
    )


def regenerate_report(
    output_dir: Path | str,
    input_dir: Path | str | None = None,
    cfg: PipelineConfig | None = None,
) -> Path:
    """Regenerate reports/summary_plots/ and reports/report.html on an existing run directory."""
    out_root = Path(output_dir)
    summary_path = out_root / "blobs_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Cannot regenerate report: missing {summary_path}")

    resolved_cfg = cfg if cfg is not None else PipelineConfig()
    summary_df = pd.read_csv(summary_path)

    qc_path = out_root / "qc" / "frame_qc.csv"
    qc_df = pd.read_csv(qc_path) if qc_path.is_file() else None

    shifts_path = out_root / "tracking" / "tracked_shifts.csv"
    shifts_df = pd.read_csv(shifts_path) if shifts_path.is_file() else pd.DataFrame()

    frames: list[FrameInput] | None = None
    if input_dir is not None:
        frames = discover_inputs(input_dir, file_order=resolved_cfg.input.file_order)

    summary_plots: list[Path] = []
    if resolved_cfg.reporting.save_summary_plots:
        summary_plots = generate_summary_plots(summary_df, shifts_df, out_root)

    return generate_html_report(
        summary_df=summary_df,
        qc_df=qc_df,
        shifts_df=shifts_df,
        output_dir=out_root,
        summary_plots=summary_plots,
        rep_cfg=resolved_cfg.reporting,
        frames=frames,
    )
