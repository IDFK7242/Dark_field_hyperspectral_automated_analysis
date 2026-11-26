"""Stage 2: Brace-aware ENVI header parser, memmap cube reader, and Z-profile extraction."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from darkfield_hsi.config import EnviConfig
from darkfield_hsi.io_utils import FrameInput

logger = logging.getLogger(__name__)

# ENVI data type codes -> NumPy base dtype strings
ENVI_DTYPE_MAP: dict[int, str] = {
    1: "u1",   # uint8
    2: "i2",   # int16
    3: "i4",   # int32
    4: "f4",   # float32
    5: "f8",   # float64
    12: "u2",  # uint16
    13: "u4",  # uint32
    14: "i8",  # int64
    15: "u8",  # uint64
}


@dataclass(frozen=True)
class EnviHeader:
    """Parsed metadata from an ENVI .hdr file."""

    samples: int
    lines: int
    bands: int
    data_type: int
    interleave: str
    byte_order: int = 0
    header_offset: int = 0
    wavelengths: np.ndarray | None = None
    pixel_size: tuple[float, ...] | None = None
    default_bands: tuple[int, ...] | None = None
    raw_metadata: dict[str, str] | None = None


def _parse_float_list(value: str) -> list[float]:
    tokens = [tok for tok in re.split(r"[,\s]+", value.strip()) if tok]
    return [float(tok) for tok in tokens]


def parse_envi_header(header_path: Path | str) -> EnviHeader:
    """Parse an ENVI .hdr file with proper multi-line brace `{ ... }` handling."""
    path = Path(header_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    metadata: dict[str, str] = {}
    current_key: str | None = None
    value_parts: list[str] = []
    brace_depth = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if current_key is None and line.upper() == "ENVI":
            continue

        if current_key is None:
            if "=" not in line:
                continue
            key_part, val_part = line.split("=", 1)
            key = key_part.strip().lower()
            val = val_part.strip()
            depth = val.count("{") - val.count("}")
            if depth > 0:
                current_key = key
                value_parts = [val]
                brace_depth = depth
            else:
                if val.startswith("{") and val.endswith("}"):
                    val = val[1:-1].strip()
                metadata[key] = val
        else:
            value_parts.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                joined = "\n".join(value_parts).strip()
                if joined.startswith("{") and joined.endswith("}"):
                    joined = joined[1:-1].strip()
                metadata[current_key] = joined
                current_key = None
                value_parts = []
                brace_depth = 0

    required = ("samples", "lines", "bands", "data type")
    missing = [k for k in required if k not in metadata]
    if missing:
        raise ValueError(f"ENVI header {path} is missing required fields: {missing}")

    samples = int(metadata["samples"])
    lines = int(metadata["lines"])
    bands = int(metadata["bands"])
    data_type = int(metadata["data type"])
    interleave = metadata.get("interleave", "bsq").strip().lower()
    byte_order = int(metadata.get("byte order", "0"))
    header_offset = int(metadata.get("header offset", "0"))

    if interleave not in ("bsq", "bil", "bip"):
        raise ValueError(f"Unsupported ENVI interleave {interleave!r} in {path}")
    if data_type not in ENVI_DTYPE_MAP:
        raise ValueError(f"Unsupported ENVI data type {data_type} in {path}")

    wavelengths: np.ndarray | None = None
    if "wavelength" in metadata and metadata["wavelength"]:
        wl_vals = _parse_float_list(metadata["wavelength"])
        if len(wl_vals) == bands:
            wavelengths = np.asarray(wl_vals, dtype=np.float64)
        else:
            logger.warning(
                "Header %s has %d wavelengths for %d bands; ignoring wavelength array.",
                path,
                len(wl_vals),
                bands,
            )

    pixel_size: tuple[float, ...] | None = None
    if "pixel size" in metadata and metadata["pixel size"]:
        try:
            pixel_size = tuple(_parse_float_list(metadata["pixel size"]))
        except ValueError:
            pixel_size = None

    default_bands: tuple[int, ...] | None = None
    if "default bands" in metadata and metadata["default bands"]:
        try:
            default_bands = tuple(int(round(v)) for v in _parse_float_list(metadata["default bands"]))
        except ValueError:
            default_bands = None

    return EnviHeader(
        samples=samples,
        lines=lines,
        bands=bands,
        data_type=data_type,
        interleave=interleave,
        byte_order=byte_order,
        header_offset=header_offset,
        wavelengths=wavelengths,
        pixel_size=pixel_size,
        default_bands=default_bands,
        raw_metadata=metadata,
    )


class EnviCube:
    """Memory-mapped reader for ENVI hyperspectral cubes (BSQ/BIL/BIP)."""

    def __init__(self, header_path: Path | str, data_path: Path | str | None = None) -> None:
        self.header_path = Path(header_path)
        self.header = parse_envi_header(self.header_path)
        self.data_path = (
            Path(data_path) if data_path is not None else self.header_path.with_suffix("")
        )
        if not self.data_path.is_file():
            raise FileNotFoundError(f"ENVI binary data file not found: {self.data_path}")

        base_dtype = ENVI_DTYPE_MAP[self.header.data_type]
        if base_dtype == "u1":
            np_dtype = np.dtype(base_dtype)
        else:
            endian = "<" if self.header.byte_order == 0 else ">"
            np_dtype = np.dtype(endian + base_dtype)

        if self.header.interleave == "bsq":
            shape = (self.header.bands, self.header.lines, self.header.samples)
        elif self.header.interleave == "bil":
            shape = (self.header.lines, self.header.bands, self.header.samples)
        else:  # bip
            shape = (self.header.lines, self.header.samples, self.header.bands)

        self._mmap = np.memmap(
            self.data_path,
            dtype=np_dtype,
            mode="r",
            offset=self.header.header_offset,
            shape=shape,
        )

    def __enter__(self) -> EnviCube:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            mmap_obj = getattr(self._mmap, "_mmap", None)
            if mmap_obj is not None:
                mmap_obj.close()
            self._mmap = None

    def get_pixel_spectrum(self, x: float, y: float) -> np.ndarray:
        """Extract single-pixel Z-profile at (x, y) clamped to image bounds."""
        ix = int(np.clip(round(x), 0, self.header.samples - 1))
        iy = int(np.clip(round(y), 0, self.header.lines - 1))
        if self.header.interleave == "bsq":
            vec = self._mmap[:, iy, ix]
        elif self.header.interleave == "bil":
            vec = self._mmap[iy, :, ix]
        else:
            vec = self._mmap[iy, ix, :]
        return np.asarray(vec, dtype=np.float64)

    def get_disk_spectrum(self, cx: float, cy: float, radius: float) -> np.ndarray:
        """Extract mean Z-profile over a circular disk centered at (cx, cy)."""
        r = max(1.0, float(radius))
        x0 = max(0, int(math.floor(cx - r)))
        x1 = min(self.header.samples, int(math.ceil(cx + r)) + 1)
        y0 = max(0, int(math.floor(cy - r)))
        y1 = min(self.header.lines, int(math.ceil(cy + r)) + 1)
        if x1 <= x0 or y1 <= y0:
            return self.get_pixel_spectrum(cx, cy)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        if not np.any(mask):
            return self.get_pixel_spectrum(cx, cy)

        if self.header.interleave == "bsq":
            patch = np.asarray(self._mmap[:, y0:y1, x0:x1], dtype=np.float64)
            return patch[:, mask].mean(axis=1)
        if self.header.interleave == "bil":
            patch = np.asarray(self._mmap[y0:y1, :, x0:x1], dtype=np.float64)
            # (dy, bands, dx) -> (bands, dy, dx)
            return np.transpose(patch, (1, 0, 2))[:, mask].mean(axis=1)
        patch = np.asarray(self._mmap[y0:y1, x0:x1, :], dtype=np.float64)
        return patch[mask].mean(axis=0)


def extract_spectrum(cube: EnviCube, x: float, y: float, radius: float, cfg: EnviConfig) -> np.ndarray:
    """Extract a blob spectrum according to EnviConfig.extraction_mode."""
    if cfg.extraction_mode == "pixel":
        return cube.get_pixel_spectrum(x, y)
    if cfg.extraction_mode == "fixed_disk":
        return cube.get_disk_spectrum(x, y, cfg.fixed_radius_px)
    if cfg.extraction_mode == "blob_disk":
        return cube.get_disk_spectrum(x, y, max(1.0, radius * cfg.blob_radius_fraction))
    raise ValueError(f"Unknown extraction_mode: {cfg.extraction_mode!r}")


def _save_z_profile_plot(
    spectrum: np.ndarray,
    wavelengths: np.ndarray | None,
    blob_id: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_axis = wavelengths if wavelengths is not None else np.arange(len(spectrum))
    x_label = "Wavelength (nm)" if wavelengths is not None else "Band"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_axis, spectrum, label=f"Blob ID: {blob_id}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Intensity")
    ax.set_title(f"Z-Profile for Blob ID {blob_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_envi_extraction(
    frames: Sequence[FrameInput],
    blobs_df: pd.DataFrame,
    output_dir: Path | str,
    envi_cfg: EnviConfig,
) -> list[Path]:
    """Extract ENVI Z-profiles for all frames that have matched hyperspectral cubes."""
    z_dir = Path(output_dir) / "z_profiles"
    z_dir.mkdir(parents=True, exist_ok=True)

    frame_by_stem = {f.stem: f for f in frames}
    written_csvs: list[Path] = []

    if blobs_df.empty:
        logger.warning("No blobs provided for ENVI extraction; skipping.")
        return written_csvs

    for stem, group in blobs_df.groupby("Filename", sort=False):
        stem_str = str(stem)
        frame = frame_by_stem.get(stem_str)
        if frame is None or not frame.has_envi:
            logger.info("No ENVI cube matched for frame %r; skipping Stage 2 for this frame.", stem_str)
            continue

        assert frame.envi_header_path is not None
        assert frame.envi_data_path is not None

        with EnviCube(frame.envi_header_path, frame.envi_data_path) as cube:
            bands = cube.header.bands
            wavelengths = cube.header.wavelengths

            plot_dir = z_dir / "plots" / stem_str
            if envi_cfg.save_plots:
                plot_dir.mkdir(parents=True, exist_ok=True)

            meta_rows: list[dict[str, object]] = []
            spectra_matrix = np.empty((len(group), bands), dtype=np.float64)

            for i, (_, row) in enumerate(group.iterrows()):
                blob_id = int(row["Blob_ID"])
                frame_num = int(row["Frame_Number"])
                x = float(row["X"])
                y = float(row["Y"])
                radius = float(row["Radius"])

                spec = extract_spectrum(cube, x, y, radius, envi_cfg)
                spectra_matrix[i] = spec
                meta_rows.append(
                    {
                        "Blob_ID": blob_id,
                        "Frame_Number": frame_num,
                        "Filename": stem_str,
                        "X": x,
                        "Y": y,
                        "Radius": radius,
                    }
                )

                if envi_cfg.save_plots:
                    _save_z_profile_plot(
                        spec,
                        wavelengths,
                        blob_id,
                        plot_dir / f"{stem_str}_b{blob_id:05d}_z_profile.png",
                    )

            meta_df = pd.DataFrame(meta_rows)
            band_cols = [f"Band_{b}" for b in range(bands)]
            bands_df = pd.DataFrame(spectra_matrix, columns=band_cols)
            out_df = pd.concat([meta_df, bands_df], axis=1)

            z_csv_path = z_dir / f"{stem_str}_z_profile.csv"
            out_df.to_csv(z_csv_path, index=False)
            written_csvs.append(z_csv_path)

            if wavelengths is not None:
                wl_df = pd.DataFrame(
                    {
                        "Band_Index": np.arange(bands, dtype=int),
                        "Wavelength_nm": wavelengths,
                    }
                )
                wl_df.to_csv(z_dir / f"{stem_str}_wavelengths.csv", index=False)

            logger.info(
                "Extracted %d Z-profile(s) (%d bands, mode=%s) -> %s",
                len(out_df),
                bands,
                envi_cfg.extraction_mode,
                z_csv_path,
            )

    return written_csvs
