"""Input directory discovery and TIFF <-> ENVI file matching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TIFF_SUFFIXES = (".tif", ".tiff")
ENVI_DATA_SUFFIXES = ("", ".raw", ".img", ".bil", ".bsq", ".bip", ".dat")


@dataclass(frozen=True)
class FrameInput:
    """Matched TIFF and optional ENVI hyperspectral pair for a single frame."""

    stem: str
    tiff_path: Path
    envi_header_path: Path | None = None
    envi_data_path: Path | None = None

    @property
    def has_envi(self) -> bool:
        return self.envi_header_path is not None and self.envi_data_path is not None


def _list_tiffs(directory: Path) -> list[Path]:
    return sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in TIFF_SUFFIXES),
        key=lambda p: p.name,
    )


def _find_envi_data_file(header_path: Path) -> Path | None:
    base = header_path.with_suffix("")
    for suffix in ENVI_DATA_SUFFIXES:
        candidate = base if suffix == "" else base.with_suffix(suffix)
        if candidate.is_file() and candidate != header_path:
            return candidate
    return None


def _match_envi(stem: str, envi_dir: Path | None) -> tuple[Path | None, Path | None]:
    if envi_dir is None or not envi_dir.is_dir():
        return None, None
    hdr = envi_dir / f"{stem}.hdr"
    if not hdr.is_file():
        return None, None
    data_path = _find_envi_data_file(hdr)
    if data_path is None:
        logger.warning("Found ENVI header %s but no companion binary data file.", hdr)
        return None, None
    return hdr, data_path


def discover_inputs(
    input_dir: Path | str,
    file_order: list[str] | None = None,
) -> list[FrameInput]:
    """Discover TIFF frames and auto-match optional ENVI pairs by filename stem.

    Supports either:
      - `<input_dir>/tiff/` + optional `<input_dir>/envi/` subfolders, or
      - co-located TIFF and ENVI files directly inside `<input_dir>`.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    tiff_subdir = root / "tiff"
    if tiff_subdir.is_dir():
        tiffs = _list_tiffs(tiff_subdir)
        tiff_dir = tiff_subdir
    else:
        tiffs = _list_tiffs(root)
        tiff_dir = root

    if not tiffs:
        raise FileNotFoundError(f"No .tif/.tiff files found in {tiff_dir}")

    envi_subdir = root / "envi"
    envi_dir = envi_subdir if envi_subdir.is_dir() else root

    by_stem: dict[str, Path] = {}
    for tiff_path in tiffs:
        if tiff_path.stem in by_stem:
            logger.warning(
                "Duplicate TIFF stem %r (%s and %s); keeping first.",
                tiff_path.stem,
                by_stem[tiff_path.stem].name,
                tiff_path.name,
            )
            continue
        by_stem[tiff_path.stem] = tiff_path

    if file_order is not None:
        ordered_stems: list[str] = []
        for stem in file_order:
            if stem not in by_stem:
                raise FileNotFoundError(
                    f"Stem {stem!r} from input.file_order not found in {tiff_dir}. "
                    f"Available: {sorted(by_stem.keys())}"
                )
            ordered_stems.append(stem)
    else:
        ordered_stems = list(by_stem.keys())

    frames: list[FrameInput] = []
    for stem in ordered_stems:
        hdr, data_path = _match_envi(stem, envi_dir)
        frames.append(
            FrameInput(
                stem=stem,
                tiff_path=by_stem[stem],
                envi_header_path=hdr,
                envi_data_path=data_path,
            )
        )

    n_envi = sum(1 for f in frames if f.has_envi)
    logger.info(
        "Discovered %d TIFF frame(s) in %s (%d matched with ENVI cubes).",
        len(frames),
        tiff_dir,
        n_envi,
    )
    return frames
