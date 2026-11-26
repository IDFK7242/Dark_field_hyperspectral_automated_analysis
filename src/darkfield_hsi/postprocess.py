"""Stage 4a: Foreground-masked circular HSV color classification and centered crops."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from darkfield_hsi.config import PostprocessConfig
from darkfield_hsi.io_utils import FrameInput

logger = logging.getLogger(__name__)

POSTPROCESS_COLUMNS = [
    "Blob_ID",
    "Frame_Number",
    "Filename",
    "Color",
    "Hue_mean_deg",
    "Hue_std_deg",
    "Sat_mean",
    "Val_mean",
    "Fg_Pixel_Count",
    "n_used_for_hue",
    "Snapshot_Normal",
    "Snapshot_Marked",
]


def is_grayscale(pil_img: Image.Image) -> bool:
    return pil_img.mode in ("L", "I;16", "I", "F")


def to_numpy_rgb(pil_img: Image.Image) -> np.ndarray:
    if is_grayscale(pil_img):
        g = np.array(pil_img.convert("L"))
        return np.stack([g, g, g], axis=-1)
    return np.array(pil_img.convert("RGB"))


def rgb_to_hsv(rgb: np.ndarray) -> tuple[float, float, float]:
    r, g, b = float(rgb[0]) / 255.0, float(rgb[1]) / 255.0, float(rgb[2]) / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    df = mx - mn
    if df < 1e-6:
        h = 0.0
    elif mx == r:
        h = (60.0 * ((g - b) / df) + 360.0) % 360.0
    elif mx == g:
        h = (60.0 * ((b - r) / df) + 120.0) % 360.0
    else:
        h = (60.0 * ((r - g) / df) + 240.0) % 360.0
    s = 0.0 if mx < 1e-6 else df / (mx + 1e-12)
    v = mx
    return h, s, v


def coarse_color_name(rgb: np.ndarray, gray: bool) -> str:
    if gray:
        v = int(round(float(rgb[0])))
        return f"Gray({v})"
    h, s, v = rgb_to_hsv(rgb)
    if v < 0.08:
        return "Black"
    if s < 0.15:
        return f"Gray({int(round(v * 255))})"
    if (h < 15.0) or (h >= 345.0):
        return "Red"
    if h < 45.0:
        return "Orange"
    if h < 70.0:
        return "Yellow"
    if h < 170.0:
        return "Green"
    if h < 200.0:
        return "Cyan"
    if h < 255.0:
        return "Blue"
    if h < 310.0:
        return "Purple"
    return "Magenta"


def _box_from_center_and_radius(
    cx: float, cy: float, r: float, width: int, height: int
) -> tuple[int, int, int, int]:
    x0 = max(0, int(math.floor(cx - r)))
    x1 = min(width, int(math.ceil(cx + r)) + 1)
    y0 = max(0, int(math.floor(cy - r)))
    y1 = min(height, int(math.ceil(cy + r)) + 1)
    return x0, x1, y0, y1


def foreground_color_stats(
    img_rgb: np.ndarray,
    cx: float,
    cy: float,
    radius_pix: float,
    cfg: PostprocessConfig,
) -> tuple[np.ndarray, float, float, float, float, int, int]:
    """Compute RGB/HSV stats strictly on foreground spot pixels (excluding dark background noise).

    Returns (mean_rgb, h_mean_deg, h_std_deg, s_mean, v_mean, fg_count, n_used_for_hue).
    """
    height, width = img_rgb.shape[:2]
    r_mask = max(1.0, math.sqrt(cfg.area_frac_for_stats) * float(radius_pix))
    r_outer = r_mask + 6.0
    x0, x1, y0, y1 = _box_from_center_and_radius(cx, cy, r_outer, width, height)
    if x1 <= x0 or y1 <= y0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0, 0.0, 0.0, 0, 0

    patch_u8 = img_rgb[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    disk = dist2 <= r_mask**2
    if not np.any(disk):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0, 0.0, 0.0, 0.0, 0, 0

    # Local background luminance from outer annulus
    annulus = (dist2 >= (r_mask + 2.0) ** 2) & (dist2 <= r_outer**2)
    val_255 = np.max(patch_u8, axis=-1)
    if np.any(annulus):
        bg_vals = val_255[annulus]
        bg_med = float(np.median(bg_vals))
        bg_mad = float(np.median(np.abs(bg_vals - bg_med)))
    else:
        bg_med, bg_mad = 0.0, 0.0

    fg_thresh = bg_med + cfg.fg_mad_mult * max(bg_mad, 2.0)
    fg_mask = disk & (val_255 >= fg_thresh)
    fg_count = int(fg_mask.sum())

    # Fall back to the top brightest pixels inside the disk if threshold is too tight
    if fg_count < cfg.fg_min_pixels:
        disk_ys, disk_xs = np.nonzero(disk)
        disk_vals = val_255[disk_ys, disk_xs]
        k = min(cfg.fg_min_pixels, len(disk_vals))
        top_idx = np.argsort(disk_vals)[-k:]
        fg_mask = np.zeros_like(disk, dtype=bool)
        fg_mask[disk_ys[top_idx], disk_xs[top_idx]] = True
        fg_count = int(fg_mask.sum())

    fg_pixels_255 = patch_u8[fg_mask]
    mean_rgb = fg_pixels_255.mean(axis=0)

    fg_norm = fg_pixels_255 / 255.0
    r, g, b = fg_norm[:, 0], fg_norm[:, 1], fg_norm[:, 2]
    mx = np.max(fg_norm, axis=-1)
    mn = np.min(fg_norm, axis=-1)
    df = mx - mn
    s = np.where(mx < 1e-6, 0.0, df / (mx + 1e-12))
    v = mx

    valid = (df > 1e-6) & (s >= cfg.sat_min_for_hue)
    n_used = int(valid.sum())
    if n_used == 0:
        return mean_rgb, 0.0, 0.0, float(s.mean()), float(v.mean()), fg_count, 0

    h = np.zeros_like(mx)
    r_eq = (mx == r) & valid
    g_eq = (mx == g) & valid
    b_eq = (mx == b) & valid
    h[r_eq] = (60.0 * ((g[r_eq] - b[r_eq]) / df[r_eq]) + 360.0) % 360.0
    h[g_eq] = (60.0 * ((b[g_eq] - r[g_eq]) / df[g_eq]) + 120.0) % 360.0
    h[b_eq] = (60.0 * ((r[b_eq] - g[b_eq]) / df[b_eq]) + 240.0) % 360.0

    ang = np.deg2rad(h[valid])
    c_mean, s_sin = float(np.cos(ang).mean()), float(np.sin(ang).mean())
    resultant = math.hypot(c_mean, s_sin)
    h_mean = (math.degrees(math.atan2(s_sin, c_mean)) + 360.0) % 360.0
    r_clip = min(max(resultant, 1e-8), 1.0)
    h_std = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(r_clip))))
    return mean_rgb, float(h_mean), float(h_std), float(s[valid].mean()), float(v[valid].mean()), fg_count, n_used


def crop_centered(pil_img: Image.Image, cx: float, cy: float, crop_size: int) -> Image.Image:
    """Extract a fixed crop_size x crop_size patch centered at (cx, cy), black-padded."""
    half = crop_size // 2
    width, height = pil_img.size
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x1), min(height, y1)
    dx0, dy0 = sx0 - x0, sy0 - y0

    mode = "L" if is_grayscale(pil_img) else "RGB"
    canvas = Image.new(mode, (crop_size, crop_size), 0)
    if sx1 > sx0 and sy1 > sy0:
        patch = pil_img.crop((sx0, sy0, sx1, sy1))
        canvas.paste(patch, (dx0, dy0))
    return canvas


def draw_crosshair(
    img: Image.Image,
    color: tuple[int, int, int] = (255, 0, 0),
    size: int = 10,
    gap: int = 4,
    thickness: int = 1,
) -> Image.Image:
    """Draw a centered split crosshair that leaves the particle center unobstructed."""
    draw = ImageDraw.Draw(img)
    cx, cy = img.width // 2, img.height // 2
    draw.line((cx - size, cy, cx - gap, cy), fill=color, width=thickness)
    draw.line((cx + gap, cy, cx + size, cy), fill=color, width=thickness)
    draw.line((cx, cy - size, cx, cy - gap), fill=color, width=thickness)
    draw.line((cx, cy + gap, cx, cy + size), fill=color, width=thickness)
    return img


def run_postprocess(
    frames: Sequence[FrameInput],
    blobs_df: pd.DataFrame,
    output_dir: Path | str,
    pp_cfg: PostprocessConfig,
) -> pd.DataFrame:
    """Run foreground-masked color classification and normal/marked crop generation."""
    out_root = Path(output_dir)
    pp_dir = out_root / "postprocess"
    normal_dir = out_root / "crops" / "normal"
    marked_dir = out_root / "crops" / "marked"
    pp_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    marked_dir.mkdir(parents=True, exist_ok=True)

    if blobs_df.empty:
        empty = pd.DataFrame(columns=POSTPROCESS_COLUMNS)
        empty.to_csv(pp_dir / "blobs_color_crops.csv", index=False)
        empty.to_csv(pp_dir / "snapshots_manifest.csv", index=False)
        return empty

    frame_by_stem = {f.stem: f for f in frames}
    records: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for stem, group in blobs_df.groupby("Filename", sort=False):
        stem_str = str(stem).strip()
        frame = frame_by_stem.get(stem_str)
        if frame is None:
            logger.warning("TIFF frame for stem %r not found; skipping postprocess.", stem_str)
            continue

        with Image.open(frame.tiff_path) as img:
            gray = is_grayscale(img)
            img_rgb = to_numpy_rgb(img)

            for _, row in group.iterrows():
                blob_id = int(row["Blob_ID"])
                frame_num = int(row["Frame_Number"]) if "Frame_Number" in group.columns else 0
                cx = float(row["X"])
                cy = float(row["Y"])
                radius = float(row["Radius"])
                radius_pix = max(radius, pp_cfg.radius_floor_px)

                mean_rgb, h_m, h_s, s_m, v_m, fg_cnt, n_used = foreground_color_stats(
                    img_rgb, cx, cy, radius_pix, pp_cfg
                )
                color_label = coarse_color_name(mean_rgb, gray)

                crop_name = f"{stem_str}_b{blob_id:05d}.png"
                snap = crop_centered(img, cx, cy, pp_cfg.crop_size)
                out_norm = normal_dir / crop_name
                snap.save(out_norm, optimize=True)

                marked = snap.copy().convert("RGB")
                draw_crosshair(
                    marked,
                    color=pp_cfg.crosshair_color,
                    size=pp_cfg.crosshair_size,
                    gap=pp_cfg.crosshair_gap,
                    thickness=pp_cfg.crosshair_thickness,
                )
                out_mark = marked_dir / crop_name
                marked.save(out_mark, optimize=True)

                rel_norm = str(out_norm.relative_to(out_root))
                rel_mark = str(out_mark.relative_to(out_root))

                rec = {
                    "Blob_ID": blob_id,
                    "Frame_Number": frame_num,
                    "Filename": stem_str,
                    "Color": color_label,
                    "Hue_mean_deg": round(h_m, 2),
                    "Hue_std_deg": round(h_s, 2),
                    "Sat_mean": round(s_m, 3),
                    "Val_mean": round(v_m, 3),
                    "Fg_Pixel_Count": int(fg_cnt),
                    "n_used_for_hue": int(n_used),
                    "Snapshot_Normal": rel_norm,
                    "Snapshot_Marked": rel_mark,
                }
                records.append(rec)
                manifest_rows.append(
                    {
                        "Filename": stem_str,
                        "Frame_Number": frame_num,
                        "Blob_ID": blob_id,
                        "X": cx,
                        "Y": cy,
                        "Radius": radius,
                        **{k: rec[k] for k in POSTPROCESS_COLUMNS[3:]},
                    }
                )

        logger.info("Post-processed %d blob(s) (foreground color + crops) for %s.", len(group), stem_str)

    pp_df = pd.DataFrame(records, columns=POSTPROCESS_COLUMNS)
    pp_csv = pp_dir / "blobs_color_crops.csv"
    manifest_csv = pp_dir / "snapshots_manifest.csv"
    pp_df.to_csv(pp_csv, index=False)
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    logger.info("Saved foreground color & crops table (%d rows) -> %s", len(pp_df), pp_csv)
    return pp_df
