"""Configuration dataclasses, material presets, YAML loading/saving, and validation."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

VALID_EXTRACTION_MODES = ("pixel", "fixed_disk", "blob_disk")
VALID_BASELINES = ("zero", "min")
VALID_PROJECTIONS = ("max_rgb", "mean_rgb", "gray")
VALID_MATERIAL_PRESETS = ("agnp", "aunp", "custom")

MATERIAL_PRESETS: dict[str, dict[str, float]] = {
    "agnp": {"red_shift_nm": 580.0, "broad_fwhm_nm": 145.0},
    "aunp": {"red_shift_nm": 620.0, "broad_fwhm_nm": 140.0},
    "custom": {"red_shift_nm": 580.0, "broad_fwhm_nm": 145.0},
}


def resolve_workers(workers: int | str) -> int:
    """Resolve 'auto' or integer worker count into a concrete positive process count."""
    if isinstance(workers, str):
        if workers.strip().lower() == "auto":
            return max(1, (os.cpu_count() or 2) - 1)
        raise ValueError(f"workers must be 'auto' or an integer >= 1, got {workers!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be 'auto' or an integer >= 1, got {workers!r}")
    return workers


@dataclass
class InputConfig:
    """Controls how input frames are ordered."""

    file_order: list[str] | None = None


@dataclass
class QcConfig:
    """Parameters for Stage 0 TIFF frame quality-control gate."""

    enabled: bool = True
    max_bg_median: float = 80.0
    max_noise_mad: float = 25.0
    max_hf_noise_mad: float = 15.0
    max_stripe_index: float = 2.5
    min_frame_snr: float = 4.0
    min_laplacian_var: float = 20.0
    use_relative_outliers: bool = True
    outlier_z_thresh: float = 3.5


@dataclass
class DetectionConfig:
    """Parameters for detection-only TIFF pre-cleaning and LoG blob detection."""

    # Detection-only TIFF pre-cleaning (original RGB TIFF remains untouched)
    clean_enabled: bool = True
    channel_projection: str = "max_rgb"  # "max_rgb" | "mean_rgb" | "gray"
    destripe_columns: bool = True
    median_ksize: int = 3
    gaussian_sigma: float = 0.7
    bg_clip_mad_mult: float = 3.0
    save_cleaned_images: bool = False

    # LoG detection parameters
    max_sigma: float = 19.0
    num_sigma: int = 16
    threshold: float = 0.04
    overlap: float = 0.01
    radius_scale: float = math.sqrt(2.5)
    min_radius: float = 1.5

    # Optional artifact filters
    border_margin_px: int = 0
    min_peak_intensity: float = 0.0
    save_tracked_images: bool = True


@dataclass
class TrackingConfig:
    """Parameters for drift-registered 1-to-1 Hungarian tracking & shift reporting."""

    enabled: bool = True
    max_distance_px: float = 4.0
    register_drift: bool = True
    drift_upsample_factor: int = 10
    save_overlay_plots: bool = False
    max_overlay_plots: int = 200


@dataclass
class EnviConfig:
    """Parameters for ENVI hyperspectral cube Z-profile extraction."""

    extraction_mode: str = "pixel"
    fixed_radius_px: float = 2.0
    blob_radius_fraction: float = 0.6
    save_plots: bool = False


@dataclass
class SpectraConfig:
    """Parameters for Savitzky-Golay smoothing, peak/multi-peak detection, FWHM, and plots."""

    smoothing_window: int = 21
    smoothing_poly: int = 3
    half_max_baseline: str = "zero"
    wavelength_min_nm: float | None = None
    wavelength_max_nm: float | None = None
    min_snr: float = 5.0
    peak_prominence_frac: float = 0.15
    peak_min_distance_bands: int = 12
    save_plots: bool = True
    plot_show_raw: bool = False
    plot_dual_band_axis: bool = True
    save_per_blob_csv: bool = True
    include_normalized_column: bool = True
    save_per_blob_metrics_csv: bool = False
    workers: int | str = "auto"


@dataclass
class PostprocessConfig:
    """Parameters for foreground-masked HSV color stats and centered crops."""

    crop_size: int = 64
    area_frac_for_stats: float = 0.6
    radius_floor_px: float = 4.0
    fg_mad_mult: float = 3.0
    fg_min_pixels: int = 4
    sat_min_for_hue: float = 0.20
    crosshair_color: tuple[int, int, int] = (255, 0, 0)
    crosshair_size: int = 10
    crosshair_gap: int = 4
    crosshair_thickness: int = 1


@dataclass
class AggregateConfig:
    """Parameters for physics-driven multi-modal aggregate classification."""

    material_preset: str = "agnp"  # "agnp" | "aunp" | "custom"
    red_shift_nm: float | None = None
    broad_fwhm_nm: float | None = None
    rel_intensity_thresh: float = 3.0
    rel_area_thresh: float = 2.5
    min_large_spot_area_px: float = 12.0
    eccentricity_thresh: float = 0.65
    min_eccentricity_area_px: float = 9.0
    v_white_min: float = 0.85
    s_white_max: float = 0.25
    h_yel_lo: float = 40.0
    h_yel_hi: float = 70.0
    s_yel_min: float = 0.30
    v_yel_min: float = 0.60
    use_proximity: bool = False
    neighbor_factor: float = 1.20
    likely_score_thresh: float = 0.30
    aggregate_score_thresh: float = 0.55

    def resolved_red_shift_nm(self) -> float:
        if self.red_shift_nm is not None:
            return float(self.red_shift_nm)
        return MATERIAL_PRESETS[self.material_preset]["red_shift_nm"]

    def resolved_broad_fwhm_nm(self) -> float:
        if self.broad_fwhm_nm is not None:
            return float(self.broad_fwhm_nm)
        return MATERIAL_PRESETS[self.material_preset]["broad_fwhm_nm"]


@dataclass
class ReportingConfig:
    """Parameters for population summary figures and interactive HTML gallery."""

    enabled: bool = True
    save_summary_plots: bool = True
    save_html_report: bool = True
    html_page_size: int = 60


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    input: InputConfig = field(default_factory=InputConfig)
    qc: QcConfig = field(default_factory=QcConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    envi: EnviConfig = field(default_factory=EnviConfig)
    spectra: SpectraConfig = field(default_factory=SpectraConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    aggregate: AggregateConfig = field(default_factory=AggregateConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def validate(self) -> None:
        """Validate parameter ranges and invariants across all sections."""
        q = self.qc
        if q.max_bg_median <= 0:
            raise ValueError(f"qc.max_bg_median must be > 0, got {q.max_bg_median}")
        if q.max_noise_mad <= 0:
            raise ValueError(f"qc.max_noise_mad must be > 0, got {q.max_noise_mad}")
        if q.max_hf_noise_mad <= 0:
            raise ValueError(f"qc.max_hf_noise_mad must be > 0, got {q.max_hf_noise_mad}")
        if q.max_stripe_index <= 0:
            raise ValueError(f"qc.max_stripe_index must be > 0, got {q.max_stripe_index}")
        if q.min_frame_snr < 0:
            raise ValueError(f"qc.min_frame_snr must be >= 0, got {q.min_frame_snr}")
        if q.min_laplacian_var < 0:
            raise ValueError(f"qc.min_laplacian_var must be >= 0, got {q.min_laplacian_var}")
        if q.outlier_z_thresh <= 0:
            raise ValueError(f"qc.outlier_z_thresh must be > 0, got {q.outlier_z_thresh}")

        d = self.detection
        if d.channel_projection not in VALID_PROJECTIONS:
            raise ValueError(
                f"detection.channel_projection must be one of {VALID_PROJECTIONS}, "
                f"got {d.channel_projection!r}"
            )
        if d.median_ksize != 0 and (d.median_ksize < 3 or d.median_ksize % 2 == 0):
            raise ValueError(
                f"detection.median_ksize must be 0 or an odd integer >= 3, got {d.median_ksize}"
            )
        if d.gaussian_sigma < 0:
            raise ValueError(f"detection.gaussian_sigma must be >= 0, got {d.gaussian_sigma}")
        if d.bg_clip_mad_mult < 0:
            raise ValueError(f"detection.bg_clip_mad_mult must be >= 0, got {d.bg_clip_mad_mult}")
        if d.max_sigma <= 0:
            raise ValueError(f"detection.max_sigma must be > 0, got {d.max_sigma}")
        if d.num_sigma < 1:
            raise ValueError(f"detection.num_sigma must be >= 1, got {d.num_sigma}")
        if d.threshold < 0:
            raise ValueError(f"detection.threshold must be >= 0, got {d.threshold}")
        if not (0.0 <= d.overlap <= 1.0):
            raise ValueError(f"detection.overlap must be in [0, 1], got {d.overlap}")
        if d.radius_scale <= 0:
            raise ValueError(f"detection.radius_scale must be > 0, got {d.radius_scale}")
        if d.min_radius < 0:
            raise ValueError(f"detection.min_radius must be >= 0, got {d.min_radius}")
        if d.border_margin_px < 0:
            raise ValueError(f"detection.border_margin_px must be >= 0, got {d.border_margin_px}")
        if d.min_peak_intensity < 0:
            raise ValueError(f"detection.min_peak_intensity must be >= 0, got {d.min_peak_intensity}")

        t = self.tracking
        if t.max_distance_px <= 0:
            raise ValueError(f"tracking.max_distance_px must be > 0, got {t.max_distance_px}")
        if t.drift_upsample_factor < 1:
            raise ValueError(
                f"tracking.drift_upsample_factor must be >= 1, got {t.drift_upsample_factor}"
            )
        if t.max_overlay_plots < 0:
            raise ValueError(f"tracking.max_overlay_plots must be >= 0, got {t.max_overlay_plots}")

        e = self.envi
        if e.extraction_mode not in VALID_EXTRACTION_MODES:
            raise ValueError(
                f"envi.extraction_mode must be one of {VALID_EXTRACTION_MODES}, "
                f"got {e.extraction_mode!r}"
            )
        if e.fixed_radius_px <= 0:
            raise ValueError(f"envi.fixed_radius_px must be > 0, got {e.fixed_radius_px}")
        if e.blob_radius_fraction <= 0:
            raise ValueError(
                f"envi.blob_radius_fraction must be > 0, got {e.blob_radius_fraction}"
            )

        s = self.spectra
        if s.smoothing_window < 3 or s.smoothing_window % 2 == 0:
            raise ValueError(
                f"spectra.smoothing_window must be an odd integer >= 3, got {s.smoothing_window}"
            )
        if s.smoothing_poly < 0:
            raise ValueError(f"spectra.smoothing_poly must be >= 0, got {s.smoothing_poly}")
        if s.smoothing_window <= s.smoothing_poly:
            raise ValueError(
                f"spectra.smoothing_window ({s.smoothing_window}) must be > "
                f"spectra.smoothing_poly ({s.smoothing_poly})"
            )
        if s.half_max_baseline not in VALID_BASELINES:
            raise ValueError(
                f"spectra.half_max_baseline must be one of {VALID_BASELINES}, "
                f"got {s.half_max_baseline!r}"
            )
        if (
            s.wavelength_min_nm is not None
            and s.wavelength_max_nm is not None
            and s.wavelength_min_nm >= s.wavelength_max_nm
        ):
            raise ValueError(
                f"spectra.wavelength_min_nm ({s.wavelength_min_nm}) must be < "
                f"spectra.wavelength_max_nm ({s.wavelength_max_nm})"
            )
        if s.min_snr < 0:
            raise ValueError(f"spectra.min_snr must be >= 0, got {s.min_snr}")
        if not (0.0 < s.peak_prominence_frac < 1.0):
            raise ValueError(
                f"spectra.peak_prominence_frac must be in (0, 1), got {s.peak_prominence_frac}"
            )
        if s.peak_min_distance_bands < 1:
            raise ValueError(
                f"spectra.peak_min_distance_bands must be >= 1, got {s.peak_min_distance_bands}"
            )
        resolve_workers(s.workers)

        p = self.postprocess
        if p.crop_size < 4:
            raise ValueError(f"postprocess.crop_size must be >= 4, got {p.crop_size}")
        if not (0.0 < p.area_frac_for_stats <= 1.0):
            raise ValueError(
                f"postprocess.area_frac_for_stats must be in (0, 1], got {p.area_frac_for_stats}"
            )
        if p.radius_floor_px <= 0:
            raise ValueError(f"postprocess.radius_floor_px must be > 0, got {p.radius_floor_px}")
        if p.fg_mad_mult < 0:
            raise ValueError(f"postprocess.fg_mad_mult must be >= 0, got {p.fg_mad_mult}")
        if p.fg_min_pixels < 1:
            raise ValueError(f"postprocess.fg_min_pixels must be >= 1, got {p.fg_min_pixels}")
        if len(p.crosshair_color) != 3 or any(not (0 <= c <= 255) for c in p.crosshair_color):
            raise ValueError(
                f"postprocess.crosshair_color must be a 3-tuple of ints in [0, 255], "
                f"got {p.crosshair_color}"
            )

        a = self.aggregate
        if a.material_preset not in VALID_MATERIAL_PRESETS:
            raise ValueError(
                f"aggregate.material_preset must be one of {VALID_MATERIAL_PRESETS}, "
                f"got {a.material_preset!r}"
            )
        if a.rel_intensity_thresh <= 0:
            raise ValueError(
                f"aggregate.rel_intensity_thresh must be > 0, got {a.rel_intensity_thresh}"
            )
        if a.rel_area_thresh <= 0:
            raise ValueError(f"aggregate.rel_area_thresh must be > 0, got {a.rel_area_thresh}")
        if not (0.0 < a.eccentricity_thresh < 1.0):
            raise ValueError(
                f"aggregate.eccentricity_thresh must be in (0, 1), got {a.eccentricity_thresh}"
            )
        if a.neighbor_factor <= 0:
            raise ValueError(f"aggregate.neighbor_factor must be > 0, got {a.neighbor_factor}")
        if not (0.0 < a.likely_score_thresh < a.aggregate_score_thresh <= 1.0):
            raise ValueError(
                f"aggregate score thresholds must satisfy 0 < likely ({a.likely_score_thresh}) "
                f"< aggregate ({a.aggregate_score_thresh}) <= 1.0"
            )

        r = self.reporting
        if r.html_page_size < 1:
            raise ValueError(f"reporting.html_page_size must be >= 1, got {r.html_page_size}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a YAML-friendly dictionary."""
        data = asdict(self)
        data["postprocess"]["crosshair_color"] = list(self.postprocess.crosshair_color)
        return data


def _update_section(section_cls: type, raw: dict[str, Any], section_name: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping")
    valid_names = {f.name for f in fields(section_cls)}
    unknown = set(raw.keys()) - valid_names
    if unknown:
        raise ValueError(
            f"Unknown keys in config section '{section_name}': {sorted(unknown)}"
        )
    kwargs = dict(raw)
    if section_cls is PostprocessConfig and "crosshair_color" in kwargs:
        kwargs["crosshair_color"] = tuple(int(c) for c in kwargs["crosshair_color"])
    return section_cls(**kwargs)


def load_config(path: Path | str | None = None) -> PipelineConfig:
    """Load a YAML configuration file merged over defaults and validate it."""
    if path is None:
        cfg = PipelineConfig()
        cfg.validate()
        return cfg

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level config must be a YAML mapping, got {type(raw).__name__}")

    section_map = {
        "input": InputConfig,
        "qc": QcConfig,
        "detection": DetectionConfig,
        "tracking": TrackingConfig,
        "envi": EnviConfig,
        "spectra": SpectraConfig,
        "postprocess": PostprocessConfig,
        "aggregate": AggregateConfig,
        "reporting": ReportingConfig,
    }
    unknown_top = set(raw.keys()) - set(section_map.keys())
    if unknown_top:
        raise ValueError(f"Unknown top-level config sections: {sorted(unknown_top)}")

    sections = {}
    for name, cls in section_map.items():
        if name in raw and raw[name] is not None:
            sections[name] = _update_section(cls, raw[name], name)
        else:
            sections[name] = cls()

    cfg = PipelineConfig(**sections)
    cfg.validate()
    return cfg


def save_config(cfg: PipelineConfig, path: Path | str) -> Path:
    """Write a PipelineConfig to YAML."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False)
    out_path.write_text(text, encoding="utf-8")
    return out_path
