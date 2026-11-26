"""Stage 3: Savitzky-Golay smoothing, SNR/multi-peak LSPR metrics, per-blob CSVs, and plots."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from darkfield_hsi.config import SpectraConfig, resolve_workers

logger = logging.getLogger(__name__)

SPECTRAL_COLUMNS = [
    "Blob_ID",
    "Frame_Number",
    "Filename",
    "Max_Band",
    "Max_Wavelength_nm",
    "Max_Intensity",
    "Half_Max_Intensity",
    "Left_Half_Band",
    "Right_Half_Band",
    "FWHM_Bands",
    "Left_Half_Wavelength_nm",
    "Right_Half_Wavelength_nm",
    "FWHM_nm",
    "Spectral_Noise",
    "Spectral_SNR",
    "Low_SNR",
    "Num_Peaks",
    "Secondary_Peak_Band",
    "Secondary_Peak_nm",
    "Secondary_Peak_Intensity",
    "Spectrum_CSV",
    "Spectrum_Plot",
]


@dataclass(frozen=True)
class SpectralMetrics:
    """Computed peak, FWHM, SNR, and multi-peak LSPR metrics for one blob spectrum."""

    Max_Band: int
    Max_Wavelength_nm: float | None
    Max_Intensity: float
    Half_Max_Intensity: float
    Left_Half_Band: int
    Right_Half_Band: int
    FWHM_Bands: float
    Left_Half_Wavelength_nm: float | None
    Right_Half_Wavelength_nm: float | None
    FWHM_nm: float | None
    Spectral_Noise: float
    Spectral_SNR: float
    Low_SNR: bool
    Num_Peaks: int
    Secondary_Peak_Band: int | None
    Secondary_Peak_nm: float | None
    Secondary_Peak_Intensity: float | None


def analyze_spectrum(
    intensities: np.ndarray,
    wavelengths: np.ndarray | None,
    cfg: SpectraConfig,
) -> tuple[SpectralMetrics, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Smooth a 1D spectrum and compute peak, FWHM, SNR, and multi-peak LSPR metrics.

    Returns (metrics, orig_indices, cropped_raw, smoothed, cropped_wavelengths).
    """
    raw = np.asarray(intensities, dtype=np.float64)
    n_total = len(raw)
    orig_indices = np.arange(n_total, dtype=int)

    wl: np.ndarray | None = None
    if wavelengths is not None:
        wl_arr = np.asarray(wavelengths, dtype=np.float64)
        if len(wl_arr) == n_total:
            wl = wl_arr

    if wl is not None and (cfg.wavelength_min_nm is not None or cfg.wavelength_max_nm is not None):
        mask = np.ones(n_total, dtype=bool)
        if cfg.wavelength_min_nm is not None:
            mask &= wl >= cfg.wavelength_min_nm
        if cfg.wavelength_max_nm is not None:
            mask &= wl <= cfg.wavelength_max_nm
        raw = raw[mask]
        orig_indices = orig_indices[mask]
        wl = wl[mask]

    if len(raw) < cfg.smoothing_window:
        raise ValueError(
            f"Spectrum length ({len(raw)}) is shorter than smoothing_window ({cfg.smoothing_window})."
        )

    smoothed = savgol_filter(raw, cfg.smoothing_window, cfg.smoothing_poly)

    local_max_idx = int(np.argmax(smoothed))
    max_intensity = float(smoothed[local_max_idx])
    max_band = int(orig_indices[local_max_idx])

    # Noise & SNR via robust MAD of high-frequency residual (raw - smoothed)
    residual = raw - smoothed
    noise = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    snr = float(max_intensity / max(noise, 1e-6)) if max_intensity > 0 else 0.0
    low_snr = bool(snr < cfg.min_snr)

    if cfg.half_max_baseline == "zero":
        half_max = max_intensity / 2.0
    elif cfg.half_max_baseline == "min":
        min_val = float(np.min(smoothed))
        half_max = min_val + (max_intensity - min_val) / 2.0
    else:
        raise ValueError(f"Unsupported half_max_baseline: {cfg.half_max_baseline!r}")

    # Left half-maximum crossing (linear sub-band interpolation)
    below_left = np.where(smoothed[:local_max_idx] <= half_max)[0]
    if len(below_left) > 0:
        i0 = int(below_left[-1])
        i1 = i0 + 1
        denom = smoothed[i1] - smoothed[i0]
        frac = float((half_max - smoothed[i0]) / denom) if abs(denom) > 1e-12 else 0.0
        left_pos = i0 + float(np.clip(frac, 0.0, 1.0))
        left_band_int = int(orig_indices[i0])
    else:
        left_pos = 0.0
        left_band_int = int(orig_indices[0])

    # Right half-maximum crossing (linear sub-band interpolation)
    below_right = np.where(smoothed[local_max_idx:] <= half_max)[0]
    if len(below_right) > 0:
        j1 = local_max_idx + int(below_right[0])
        if j1 > local_max_idx:
            j0 = j1 - 1
            denom = smoothed[j1] - smoothed[j0]
            frac = float((half_max - smoothed[j0]) / denom) if abs(denom) > 1e-12 else 0.0
            right_pos = j0 + float(np.clip(frac, 0.0, 1.0))
        else:
            right_pos = float(j1)
        right_band_int = int(orig_indices[j1])
    else:
        right_pos = float(len(smoothed) - 1)
        right_band_int = int(orig_indices[-1])

    fwhm_bands = max(0.0, float(right_pos - left_pos))

    # Multi-peak LSPR detection
    prominence = max(cfg.peak_prominence_frac * max(max_intensity, 1e-6), 3.0 * noise)
    peaks, _ = find_peaks(
        smoothed,
        prominence=prominence,
        distance=cfg.peak_min_distance_bands,
        height=max(half_max * 0.5, 3.0 * noise),
    )
    if len(peaks) > 0:
        order = np.argsort(smoothed[peaks])[::-1]
        sorted_peaks = peaks[order]
        num_peaks = int(len(sorted_peaks))
    else:
        sorted_peaks = np.array([], dtype=int)
        num_peaks = 0 if low_snr else 1

    sec_band: int | None = None
    sec_wl: float | None = None
    sec_int: float | None = None
    if len(sorted_peaks) >= 2:
        sec_idx = int(sorted_peaks[1])
        sec_band = int(orig_indices[sec_idx])
        sec_int = float(smoothed[sec_idx])
        if wl is not None:
            sec_wl = float(wl[sec_idx])

    if wl is not None:
        idx_grid = np.arange(len(wl), dtype=np.float64)
        max_wl: float | None = float(wl[local_max_idx])
        left_wl: float | None = float(np.interp(left_pos, idx_grid, wl))
        right_wl: float | None = float(np.interp(right_pos, idx_grid, wl))
        fwhm_nm: float | None = abs(right_wl - left_wl)
    else:
        max_wl = None
        left_wl = None
        right_wl = None
        fwhm_nm = None

    metrics = SpectralMetrics(
        Max_Band=max_band,
        Max_Wavelength_nm=max_wl,
        Max_Intensity=max_intensity,
        Half_Max_Intensity=float(half_max),
        Left_Half_Band=left_band_int,
        Right_Half_Band=right_band_int,
        FWHM_Bands=fwhm_bands,
        Left_Half_Wavelength_nm=left_wl,
        Right_Half_Wavelength_nm=right_wl,
        FWHM_nm=fwhm_nm,
        Spectral_Noise=round(noise, 4),
        Spectral_SNR=round(snr, 3),
        Low_SNR=low_snr,
        Num_Peaks=num_peaks,
        Secondary_Peak_Band=sec_band,
        Secondary_Peak_nm=sec_wl,
        Secondary_Peak_Intensity=sec_int,
    )
    return metrics, orig_indices, raw, smoothed, wl


def _save_blob_spectrum_csv(
    orig_indices: np.ndarray,
    raw: np.ndarray,
    smoothed: np.ndarray,
    wavelengths: np.ndarray | None,
    max_intensity: float,
    include_normalized: bool,
    out_path: Path,
) -> None:
    data: dict[str, Any] = {"Band_Index": orig_indices}
    if wavelengths is not None:
        data["Wavelength_nm"] = np.round(wavelengths, 4)
    data["Raw_Intensity"] = np.round(raw, 4)
    data["Smoothed_Intensity"] = np.round(smoothed, 4)
    if include_normalized:
        norm = smoothed / max_intensity if max_intensity > 1e-12 else np.zeros_like(smoothed)
        data["Normalized_Intensity"] = np.round(norm, 6)
    pd.DataFrame(data).to_csv(out_path, index=False)


def _save_blob_spectrum_plot(
    orig_indices: np.ndarray,
    raw: np.ndarray,
    smoothed: np.ndarray,
    wavelengths: np.ndarray | None,
    metrics: SpectralMetrics,
    blob_id: int,
    cfg: SpectraConfig,
    out_path: Path,
) -> None:
    """Save a per-blob smoothed spectrum plot modeled on Blob_999.0_smoothed.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_nm = wavelengths is not None
    x_axis = wavelengths if use_nm else orig_indices.astype(np.float64)
    x_label = "Wavelength (nm)" if use_nm else "Band"

    if use_nm and metrics.Max_Wavelength_nm is not None:
        peak_x = metrics.Max_Wavelength_nm
        left_x = metrics.Left_Half_Wavelength_nm
        right_x = metrics.Right_Half_Wavelength_nm
        peak_lbl = f"Max: {metrics.Max_Wavelength_nm:.1f} nm (Band {metrics.Max_Band})"
        left_lbl = f"Left FWHM ({left_x:.1f} nm)" if left_x is not None else "Left FWHM"
        right_lbl = (
            f"Right FWHM ({right_x:.1f} nm | FWHM: {metrics.FWHM_nm:.1f} nm)"
            if (right_x is not None and metrics.FWHM_nm is not None)
            else "Right FWHM"
        )
    else:
        peak_x = float(metrics.Max_Band)
        left_x = float(metrics.Left_Half_Band)
        right_x = float(metrics.Right_Half_Band)
        peak_lbl = f"Max Band ({metrics.Max_Band})"
        left_lbl = f"Left Half Band ({metrics.Left_Half_Band})"
        right_lbl = f"Right Half Band ({metrics.Right_Half_Band} | FWHM: {metrics.FWHM_Bands:.1f} bands)"

    fig, ax = plt.subplots(figsize=(10, 5.8))
    if cfg.plot_show_raw:
        ax.plot(x_axis, raw, color="steelblue", alpha=0.35, linewidth=0.9, label="Raw Data")
    ax.plot(x_axis, smoothed, color="green", linewidth=1.6, label=f"Smoothed Data (Blob ID: {blob_id})")
    ax.axhline(
        y=metrics.Half_Max_Intensity,
        color="red",
        linestyle="--",
        linewidth=1.3,
        label=f"Half Max Intensity ({metrics.Half_Max_Intensity:.1f})",
    )
    if left_x is not None:
        ax.axvline(x=left_x, color="green", linestyle="--", linewidth=1.3, label=left_lbl)
    if right_x is not None:
        ax.axvline(x=right_x, color="purple", linestyle="--", linewidth=1.3, label=right_lbl)
    ax.axvline(x=peak_x, color="orange", linestyle="--", linewidth=1.4, label=peak_lbl)

    if metrics.Secondary_Peak_nm is not None and use_nm:
        ax.scatter(
            [metrics.Secondary_Peak_nm],
            [metrics.Secondary_Peak_Intensity],
            color="darkorange",
            marker="o",
            s=36,
            zorder=5,
            label=f"2nd Peak ({metrics.Secondary_Peak_nm:.1f} nm)",
        )

    ax.set_title(f"Smoothed Z-Profile for Blob ID {blob_id}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Intensity")
    ax.grid(True, alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    if use_nm and cfg.plot_dual_band_axis and wavelengths is not None:
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        n_ticks = 6
        band_positions = np.linspace(0, len(orig_indices) - 1, n_ticks)
        wl_ticks = np.interp(band_positions, np.arange(len(wavelengths)), wavelengths)
        band_labels = [str(int(round(np.interp(p, np.arange(len(orig_indices)), orig_indices)))) for p in band_positions]
        ax_top.set_xticks(wl_ticks)
        ax_top.set_xticklabels(band_labels)
        ax_top.set_xlabel("Band")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _process_blob_chunk(
    args: tuple[
        list[tuple[int, int, str, np.ndarray]],
        np.ndarray | None,
        str,
        Path,
        Path,
        Path,
        Path,
        Path,
        SpectraConfig,
    ],
) -> list[dict[str, object]]:
    (
        chunk_items,
        wavelengths,
        stem,
        out_root,
        plot_dir,
        per_blob_csv_dir,
        per_blob_metrics_dir,
        _spectra_dir,
        cfg,
    ) = args

    records: list[dict[str, object]] = []
    for blob_id, frame_num, filename, intensities in chunk_items:
        metrics, orig_idx, raw_crop, smoothed, wl_crop = analyze_spectrum(intensities, wavelengths, cfg)

        spec_csv_rel = ""
        if cfg.save_per_blob_csv:
            csv_path = per_blob_csv_dir / f"{stem}_b{blob_id:05d}_spectrum.csv"
            _save_blob_spectrum_csv(
                orig_idx,
                raw_crop,
                smoothed,
                wl_crop,
                metrics.Max_Intensity,
                cfg.include_normalized_column,
                csv_path,
            )
            spec_csv_rel = str(csv_path.relative_to(out_root))

        spec_plot_rel = ""
        if cfg.save_plots:
            plot_path = plot_dir / f"{stem}_b{blob_id:05d}_smoothed.png"
            _save_blob_spectrum_plot(
                orig_idx,
                raw_crop,
                smoothed,
                wl_crop,
                metrics,
                blob_id,
                cfg,
                plot_path,
            )
            spec_plot_rel = str(plot_path.relative_to(out_root))

        rec: dict[str, object] = {
            "Blob_ID": blob_id,
            "Frame_Number": frame_num,
            "Filename": filename,
            **asdict(metrics),
            "Spectrum_CSV": spec_csv_rel,
            "Spectrum_Plot": spec_plot_rel,
        }
        records.append(rec)

        if cfg.save_per_blob_metrics_csv:
            m_path = per_blob_metrics_dir / f"Blob_{blob_id:05d}_metrics.csv"
            pd.DataFrame([rec]).to_csv(m_path, index=False)

    return records


def _process_single_z_profile_csv(
    z_csv_path: Path,
    out_root: Path,
    spectra_dir: Path,
    cfg: SpectraConfig,
    n_workers: int,
) -> pd.DataFrame:
    df = pd.read_csv(z_csv_path)
    band_cols = [c for c in df.columns if c.startswith("Band_")]
    if not band_cols:
        raise ValueError(f"No Band_* columns found in {z_csv_path}")

    stem = z_csv_path.stem
    if stem.endswith("_z_profile"):
        stem = stem[: -len("_z_profile")]

    wl_path = z_csv_path.parent / f"{stem}_wavelengths.csv"
    wavelengths: np.ndarray | None = None
    if wl_path.is_file():
        wl_df = pd.read_csv(wl_path)
        if "Wavelength_nm" in wl_df.columns and len(wl_df) == len(band_cols):
            wavelengths = wl_df["Wavelength_nm"].to_numpy(dtype=np.float64)

    plot_dir = spectra_dir / "plots" / stem
    per_blob_csv_dir = spectra_dir / "per_blob_csv" / stem
    per_blob_metrics_dir = spectra_dir / "per_blob_metrics" / stem
    if cfg.save_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_per_blob_csv:
        per_blob_csv_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_per_blob_metrics_csv:
        per_blob_metrics_dir.mkdir(parents=True, exist_ok=True)

    band_matrix = df[band_cols].to_numpy(dtype=np.float64)
    blob_ids = df["Blob_ID"].astype(int).tolist()
    frame_nums = (
        df["Frame_Number"].astype(int).tolist()
        if "Frame_Number" in df.columns
        else [0] * len(df)
    )
    filenames = (
        df["Filename"].astype(str).tolist()
        if "Filename" in df.columns
        else [stem] * len(df)
    )

    items = [
        (blob_ids[i], frame_nums[i], filenames[i], band_matrix[i])
        for i in range(len(df))
    ]

    if n_workers > 1 and len(items) >= 20 and (cfg.save_plots or cfg.save_per_blob_csv):
        n_chunks = min(len(items), n_workers * 4)
        chunk_size = max(1, (len(items) + n_chunks - 1) // n_chunks)
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
        tasks = [
            (
                chunk,
                wavelengths,
                stem,
                out_root,
                plot_dir,
                per_blob_csv_dir,
                per_blob_metrics_dir,
                spectra_dir,
                cfg,
            )
            for chunk in chunks
        ]
        all_records: list[dict[str, object]] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for part in pool.map(_process_blob_chunk, tasks):
                all_records.extend(part)
    else:
        all_records = _process_blob_chunk(
            (
                items,
                wavelengths,
                stem,
                out_root,
                plot_dir,
                per_blob_csv_dir,
                per_blob_metrics_dir,
                spectra_dir,
                cfg,
            )
        )

    result_df = pd.DataFrame(all_records, columns=SPECTRAL_COLUMNS)
    per_frame_csv = spectra_dir / f"{stem}_spectral_metrics.csv"
    result_df.to_csv(per_frame_csv, index=False)
    logger.info(
        "Analyzed %d spectrum/spectra for %s (workers=%d, plots=%s, per_blob_csv=%s) -> %s",
        len(result_df),
        stem,
        n_workers,
        cfg.save_plots,
        cfg.save_per_blob_csv,
        per_frame_csv,
    )
    return result_df


def run_spectral_analysis(
    z_profile_dir: Path | str,
    output_dir: Path | str,
    spec_cfg: SpectraConfig,
) -> pd.DataFrame:
    """Analyze all *_z_profile.csv files in z_profile_dir and write aggregated metrics."""
    z_dir = Path(z_profile_dir)
    out_root = Path(output_dir)
    spectra_dir = out_root / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    z_csvs = sorted(z_dir.glob("*_z_profile.csv"))
    if not z_csvs:
        logger.warning("No *_z_profile.csv files found in %s; skipping Stage 3.", z_dir)
        empty_df = pd.DataFrame(columns=SPECTRAL_COLUMNS)
        empty_df.to_csv(spectra_dir / "spectral_metrics.csv", index=False)
        return empty_df

    n_workers = resolve_workers(spec_cfg.workers)
    dfs = [
        _process_single_z_profile_csv(p, out_root, spectra_dir, spec_cfg, n_workers)
        for p in z_csvs
    ]

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=SPECTRAL_COLUMNS)
    combined_path = spectra_dir / "spectral_metrics.csv"
    combined.to_csv(combined_path, index=False)
    logger.info("Saved aggregated spectral metrics (%d rows) -> %s", len(combined), combined_path)
    return combined
