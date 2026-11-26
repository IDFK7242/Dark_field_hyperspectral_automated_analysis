"""Standalone zero-server interactive HTML report & fullscreen blob inspector builder."""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

from darkfield_hsi.config import ReportingConfig
from darkfield_hsi.io_utils import FrameInput

logger = logging.getLogger(__name__)


def export_fov_pngs(
    frames: Sequence[FrameInput] | None,
    summary_df: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, dict[str, object]]:
    """Export browser-readable RGB PNG copies of each dark-field frame under reports/fov/.

    Returns a mapping `{stem: {"src": relative_to_reports_dir, "width": W, "height": H}}`.
    """
    out_root = Path(output_dir)
    reports_dir = out_root / "reports"
    fov_dir = reports_dir / "fov"
    fov_dir.mkdir(parents=True, exist_ok=True)

    fov_meta: dict[str, dict[str, object]] = {}
    stems = (
        sorted({str(s) for s in summary_df["Filename"].dropna().unique()})
        if "Filename" in summary_df.columns
        else []
    )
    frame_by_stem = {f.stem: f for f in frames} if frames else {}

    for stem in stems:
        out_png = fov_dir / f"{stem}.png"
        frame = frame_by_stem.get(stem)
        if frame is not None and frame.tiff_path.is_file():
            with Image.open(frame.tiff_path) as im:
                rgb = im.convert("RGB")
                width, height = rgb.size
                rgb.save(out_png, optimize=True)
            fov_meta[stem] = {"src": f"fov/{stem}.png", "width": width, "height": height}
            continue

        # Reuse previously exported FOV PNG when regenerating without --input-dir
        if out_png.is_file():
            with Image.open(out_png) as im:
                width, height = im.size
            fov_meta[stem] = {"src": f"fov/{stem}.png", "width": width, "height": height}
            continue

        # Fallback: look for annotated tracked image
        tracked_dir = out_root / "tracked_images"
        if tracked_dir.is_dir():
            matches = sorted(tracked_dir.glob(f"*_{stem}_tracked.png"))
            if matches:
                with Image.open(matches[0]) as im:
                    width, height = im.size
                rel = f"../tracked_images/{matches[0].name}"
                fov_meta[stem] = {"src": rel, "width": width, "height": height}

    return fov_meta


def _serialize_blobs(summary_df: pd.DataFrame) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    cols = set(summary_df.columns)

    for _, row in summary_df.iterrows():

        def _val(col: str, default: object = None, digits: int = 2) -> object:
            if col not in cols:
                return default
            v = row[col]
            if pd.isna(v):
                return default
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return round(float(v), digits)
            if isinstance(v, (np.bool_,)):
                return bool(v)
            return v

        norm_rel = str(_val("Snapshot_Normal", "") or "")
        mark_rel = str(_val("Snapshot_Marked", "") or "")
        plot_rel = str(_val("Spectrum_Plot", "") or "")
        csv_rel = str(_val("Spectrum_CSV", "") or "")

        items.append(
            {
                "id": int(row["Blob_ID"]),
                "frame": int(row["Frame_Number"]),
                "file": str(row["Filename"]),
                "x": round(float(row["X"]), 1),
                "y": round(float(row["Y"]), 1),
                "radius": _val("Radius", None),
                "spot_peak": _val("Spot_Peak_Intensity", None, 1),
                "area": _val("Spot_Area_px", None),
                "spot_r": _val("Spot_Radius_FWHM_px", None),
                "ecc": _val("Eccentricity", None, 3),
                "sat_px": bool(_val("Is_Saturated", False)),
                "color": str(_val("Color", "")),
                "hue": _val("Hue_mean_deg", None, 1),
                "sat": _val("Sat_mean", None, 3),
                "val": _val("Val_mean", None, 3),
                "fg_px": _val("Fg_Pixel_Count", None),
                "agg_class": str(_val("Aggregate_Class", "Single")),
                "agg_score": _val("Aggregate_Score", 0.0, 3),
                "evidence": str(_val("Aggregate_Evidence", "Single")),
                "peak_nm": _val("Max_Wavelength_nm", None, 1),
                "peak_band": _val("Max_Band", None),
                "fwhm_nm": _val("FWHM_nm", None, 1),
                "max_int": _val("Max_Intensity", None, 1),
                "snr": _val("Spectral_SNR", None, 1),
                "low_snr": bool(_val("Low_SNR", False)),
                "num_peaks": _val("Num_Peaks", None),
                "sec_peak_nm": _val("Secondary_Peak_nm", None, 1),
                "norm": f"../{norm_rel}" if norm_rel else "",
                "mark": f"../{mark_rel}" if mark_rel else "",
                "spec": f"../{plot_rel}" if plot_rel else "",
                "spec_csv": f"../{csv_rel}" if csv_rel else "",
            }
        )
    return items


def build_html_report(
    summary_df: pd.DataFrame,
    qc_df: pd.DataFrame | None,
    shifts_df: pd.DataFrame | None,
    output_dir: Path | str,
    summary_plots: list[Path],
    rep_cfg: ReportingConfig,
    frames: Sequence[FrameInput] | None = None,
) -> Path:
    """Generate the standalone interactive reports/report.html workstation."""
    out_root = Path(output_dir)
    reports_dir = out_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "report.html"

    fov_meta = export_fov_pngs(frames, summary_df, out_root)
    blob_items = _serialize_blobs(summary_df)

    frames_list = sorted({str(item["file"]) for item in blob_items})
    colors_list = sorted({str(item["color"]) for item in blob_items if item["color"]})
    evidence_tags: set[str] = set()
    for item in blob_items:
        ev = str(item["evidence"] or "")
        for tag in ev.split("+"):
            tag = tag.strip()
            if tag and tag != "Single":
                evidence_tags.add(tag)

    n_total = len(summary_df)
    n_agg = (
        int((summary_df["Aggregate_Class"] == "Aggregate").sum())
        if "Aggregate_Class" in summary_df.columns
        else 0
    )
    n_likely = (
        int((summary_df["Aggregate_Class"] == "Likely_Aggregate").sum())
        if "Aggregate_Class" in summary_df.columns
        else 0
    )
    n_single = (
        int((summary_df["Aggregate_Class"] == "Single").sum())
        if "Aggregate_Class" in summary_df.columns
        else n_total
    )
    n_tracked = len(shifts_df) if shifts_df is not None else 0
    med_peak = (
        f"{summary_df['Max_Wavelength_nm'].median():.1f} nm"
        if ("Max_Wavelength_nm" in summary_df.columns and summary_df["Max_Wavelength_nm"].notna().any())
        else "N/A"
    )
    med_fwhm = (
        f"{summary_df['FWHM_nm'].median():.1f} nm"
        if ("FWHM_nm" in summary_df.columns and summary_df["FWHM_nm"].notna().any())
        else "N/A"
    )

    qc_section_html = ""
    if qc_df is not None and not qc_df.empty:
        qc_rows = []
        for _, qrow in qc_df.iterrows():
            status_cls = "badge-pass" if bool(qrow["Passed_QC"]) else "badge-fail"
            status_txt = "PASS" if bool(qrow["Passed_QC"]) else "REJECT"
            qc_rows.append(
                f"<tr><td>{html.escape(str(qrow['Filename']))}</td>"
                f"<td><span class='badge {status_cls}'>{status_txt}</span></td>"
                f"<td>{qrow['Bg_Median']}</td><td>{qrow['Bg_MAD']}</td>"
                f"<td>{qrow['HF_Noise_MAD']}</td><td>{qrow['Stripe_Index']}</td>"
                f"<td>{qrow['SNR_P995']}</td><td>{qrow['Laplacian_Focus']}</td>"
                f"<td>{html.escape(str(qrow['Reject_Reason']))}</td></tr>"
            )
        qc_section_html = (
            "<h2>Stage 0 — Frame Quality Control</h2>"
            "<table><thead><tr><th>Filename</th><th>Status</th><th>Bg Median</th><th>Bg MAD</th>"
            "<th>HF Noise</th><th>Stripe Idx</th><th>SNR</th><th>Focus</th><th>Reason</th></tr></thead>"
            f"<tbody>{''.join(qc_rows)}</tbody></table>"
        )

    summary_section_html = ""
    if summary_plots:
        imgs = "".join(
            f"<div class='plot-card' data-zoom-src='summary_plots/{html.escape(p.name)}' "
            f"data-zoom-title='{html.escape(p.stem.replace('_', ' ').title())}'>"
            f"<img src='summary_plots/{html.escape(p.name)}' alt='{html.escape(p.stem)}' loading='lazy'/></div>"
            for p in summary_plots
        )
        summary_section_html = (
            "<h2>Population Summary Figures <span class='hint'>(click any chart to zoom)</span></h2>"
            f"<div class='plot-grid'>{imgs}</div>"
        )

    frame_opts = "".join(f"<option value='{html.escape(f)}'>{html.escape(f)}</option>" for f in frames_list)
    color_opts = "".join(f"<option value='{html.escape(c)}'>{html.escape(c)}</option>" for c in colors_list)
    ev_opts = "".join(f"<option value='{html.escape(t)}'>{html.escape(t)}</option>" for t in sorted(evidence_tags))

    blob_json = json.dumps(blob_items, separators=(",", ":")).replace("</", "<\\/")
    fov_json = json.dumps(fov_meta, separators=(",", ":")).replace("</", "<\\/")

    html_doc = _HTML_TEMPLATE
    replacements = {
        "__TITLE__": html.escape(out_root.name),
        "__N_TOTAL__": str(n_total),
        "__N_SINGLE__": str(n_single),
        "__N_LIKELY__": str(n_likely),
        "__N_AGG__": str(n_agg),
        "__N_TRACKED__": str(n_tracked),
        "__MED_PEAK__": html.escape(med_peak),
        "__MED_FWHM__": html.escape(med_fwhm),
        "__QC_SECTION__": qc_section_html,
        "__SUMMARY_SECTION__": summary_section_html,
        "__FRAME_OPTS__": frame_opts,
        "__COLOR_OPTS__": color_opts,
        "__EVIDENCE_OPTS__": ev_opts,
        "__PAGE_SIZE__": str(int(rep_cfg.html_page_size)),
        "__BLOB_JSON__": blob_json,
        "__FOV_JSON__": fov_json,
    }
    for key, val in replacements.items():
        html_doc = html_doc.replace(key, val)

    report_path.write_text(html_doc, encoding="utf-8")
    logger.info(
        "Generated interactive HTML workstation (%d blobs, %d FOV PNGs) -> %s",
        len(blob_items),
        len(fov_meta),
        report_path,
    )
    return report_path


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>darkfield-hsi Workstation — __TITLE__</title>
<style>
  :root {
    --bg:#0d1017; --card:#161b26; --card2:#1c2333; --border:#2b3345;
    --text:#e6eaf2; --muted:#95a0b5; --accent:#4ea8de;
    --green:#2ecc71; --amber:#f39c12; --red:#e74c3c;
  }
  * { box-sizing:border-box; }
  body { margin:0; padding:22px 26px 60px; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); }
  h1,h2,h3 { margin:0 0 12px; font-weight:600; }
  h2 { margin-top:26px; font-size:1.1rem; color:#d0d8ea; border-bottom:1px solid var(--border); padding-bottom:6px; display:flex; align-items:center; gap:10px; }
  .hint { font-size:0.78rem; color:var(--muted); font-weight:400; }
  .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; margin-bottom:18px; }
  .kpi { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px 14px; }
  .kpi .lbl { font-size:0.74rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.04em; }
  .kpi .val { font-size:1.38rem; font-weight:700; margin-top:4px; }

  table { width:100%; border-collapse:collapse; background:var(--card); border-radius:8px; overflow:hidden; font-size:0.83rem; }
  th,td { padding:7px 10px; border-bottom:1px solid var(--border); text-align:left; }
  th { background:var(--card2); color:var(--muted); font-weight:600; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.74rem; font-weight:600; white-space:nowrap; }
  .badge-pass,.cls-Single { background:rgba(46,204,113,0.18); color:var(--green); }
  .cls-Likely_Aggregate { background:rgba(243,156,18,0.2); color:var(--amber); }
  .badge-fail,.cls-Aggregate { background:rgba(231,76,60,0.22); color:var(--red); }
  .ev-chip { display:inline-block; padding:2px 7px; border-radius:6px; background:#232b3d; color:#cbd5e1; font-size:0.72rem; margin:2px 4px 2px 0; border:1px solid var(--border); }

  .plot-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(390px,1fr)); gap:12px; }
  .plot-card { background:#fff; border-radius:10px; padding:8px; overflow:hidden; cursor:zoom-in; transition:transform 0.12s; }
  .plot-card:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(0,0,0,0.35); }
  .plot-card img { width:100%; height:auto; display:block; }

  /* Interactive Canvas Exploration Row */
  .explore-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:14px; margin-bottom:16px; }
  .panel { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:8px; }
  .panel-head { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; font-size:0.88rem; }
  .panel-head select { background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:4px 8px; font-size:0.8rem; }
  .canvas-wrap { position:relative; width:100%; background:#080a0f; border-radius:8px; border:1px solid var(--border); overflow:hidden; display:flex; justify-content:center; align-items:center; }
  canvas.interactive { display:block; cursor:crosshair; max-width:100%; }
  .canvas-tip { position:fixed; pointer-events:none; z-index:1500; background:rgba(15,19,29,0.95); border:1px solid var(--accent); color:var(--text); padding:6px 9px; border-radius:6px; font-size:0.77rem; line-height:1.35; display:none; box-shadow:0 4px 14px rgba(0,0,0,0.5); }

  /* Toolbar */
  .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px; margin-bottom:10px; }
  .toolbar select,.toolbar input[type="text"],.toolbar input[type="number"] { background:var(--bg); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:6px 8px; font-size:0.82rem; }
  .toolbar label.chk { display:inline-flex; align-items:center; gap:5px; font-size:0.82rem; color:var(--text); padding:4px 8px; background:var(--bg); border:1px solid var(--border); border-radius:6px; cursor:pointer; }
  .btn { background:var(--accent); color:#09111f; border:none; border-radius:6px; padding:6px 11px; font-weight:600; cursor:pointer; font-size:0.82rem; }
  .btn.secondary { background:#273044; color:var(--text); border:1px solid var(--border); }
  .btn:hover { filter:brightness(1.08); }
  .btn:disabled { opacity:0.4; cursor:not-allowed; }
  .toast { font-size:0.8rem; color:var(--green); margin-left:6px; }

  .pager { display:flex; justify-content:space-between; align-items:center; margin:10px 0 12px; font-size:0.86rem; color:var(--muted); flex-wrap:wrap; gap:8px; }
  .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(350px,1fr)); gap:13px; }
  .blob-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:11px; display:flex; flex-direction:column; gap:8px; cursor:pointer; transition:border-color 0.12s, transform 0.12s; }
  .blob-card:hover { border-color:var(--accent); transform:translateY(-2px); }
  .blob-head { display:flex; justify-content:space-between; align-items:center; gap:8px; font-size:0.87rem; }
  .crops-row { display:flex; gap:8px; align-items:center; }
  .crops-row img.crop { width:72px; height:72px; image-rendering:pixelated; border-radius:6px; border:1px solid var(--border); background:#000; }
  .spec-img { width:100%; border-radius:6px; background:#fff; display:block; }
  .meta { font-size:0.76rem; color:var(--muted); display:grid; grid-template-columns:1fr 1fr; gap:2px 8px; }
  .review-row { display:flex; gap:6px; margin-top:2px; }
  .review-btn { flex:1; padding:5px 6px; border-radius:6px; border:1px solid var(--border); background:var(--bg); color:var(--muted); cursor:pointer; font-size:0.76rem; font-weight:600; }
  .review-btn.active-Single { background:var(--green); color:#09111f; border-color:var(--green); }
  .review-btn.active-Aggregate { background:var(--red); color:#fff; border-color:var(--red); }
  .review-btn.active-Noise { background:var(--amber); color:#09111f; border-color:var(--amber); }

  /* Full-Screen Blob Inspector Modal */
  .modal-backdrop { position:fixed; inset:0; background:rgba(6,8,13,0.92); z-index:1000; display:none; align-items:center; justify-content:center; padding:18px; }
  .modal-backdrop.open { display:flex; }
  .modal-nav { position:fixed; top:50%; transform:translateY(-50%); z-index:1020; width:46px; height:74px; border-radius:10px; border:1px solid var(--border); background:rgba(22,27,38,0.92); color:var(--text); font-size:2rem; line-height:1; cursor:pointer; display:flex; align-items:center; justify-content:center; }
  .modal-nav:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
  .modal-nav:disabled { opacity:0.25; cursor:not-allowed; }
  #modal-prev { left:14px; }
  #modal-next { right:14px; }
  .modal-box { width:min(1420px, calc(100vw - 130px)); max-height:93vh; background:var(--card); border:1px solid var(--border); border-radius:14px; display:flex; flex-direction:column; overflow:hidden; box-shadow:0 16px 50px rgba(0,0,0,0.65); }
  .modal-header { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:12px 18px; background:var(--card2); border-bottom:1px solid var(--border); flex-wrap:wrap; }
  .modal-header-left { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
  .modal-pos { font-size:0.84rem; color:var(--muted); background:var(--bg); padding:3px 10px; border-radius:999px; border:1px solid var(--border); }
  .modal-body { display:grid; grid-template-columns:1.35fr 1fr; gap:16px; padding:16px 18px; overflow-y:auto; align-items:start; }
  @media (max-width: 1020px) { .modal-body { grid-template-columns:1fr; } .modal-box { width:calc(100vw - 96px); } }
  .modal-left { display:flex; flex-direction:column; gap:10px; }
  .modal-spec-wrap { background:#fff; border-radius:10px; padding:6px; overflow:hidden; }
  .modal-spec-wrap img { width:100%; height:auto; display:block; }
  .modal-links { display:flex; gap:10px; flex-wrap:wrap; align-items:center; font-size:0.82rem; }
  .modal-links a { color:var(--accent); text-decoration:none; padding:5px 10px; border:1px solid var(--border); border-radius:6px; background:var(--bg); }
  .modal-links a:hover { border-color:var(--accent); }
  .modal-right { display:flex; flex-direction:column; gap:12px; }
  .modal-visuals { display:flex; flex-wrap:wrap; gap:12px; align-items:flex-start; }
  .crop-box { display:flex; flex-direction:column; gap:5px; align-items:center; font-size:0.75rem; color:var(--muted); }
  .crop-box img { image-rendering:pixelated; border-radius:8px; border:1px solid var(--border); background:#000; }
  .zoom-bar { display:flex; gap:5px; align-items:center; font-size:0.76rem; color:var(--muted); }
  .zoom-btn { padding:3px 8px; border-radius:5px; border:1px solid var(--border); background:var(--bg); color:var(--text); cursor:pointer; font-size:0.75rem; }
  .zoom-btn.active { background:var(--accent); color:#09111f; border-color:var(--accent); font-weight:700; }
  .minimap-box { display:flex; flex-direction:column; gap:4px; align-items:center; font-size:0.74rem; color:var(--muted); }
  #modal-fov-canvas { border-radius:8px; border:1px solid var(--border); background:#000; display:block; }
  .detail-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:6px 12px; background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:10px 12px; font-size:0.82rem; }
  .detail-item { display:flex; justify-content:space-between; gap:8px; border-bottom:1px dashed rgba(43,51,69,0.5); padding-bottom:3px; }
  .detail-item span:first-child { color:var(--muted); }
  .modal-review { background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:10px; }
  .modal-review input[type="text"] { width:100%; background:var(--card); color:var(--text); border:1px solid var(--border); border-radius:6px; padding:7px 9px; font-size:0.84rem; }
  .kbd-hint { font-size:0.74rem; color:var(--muted); display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; }
  kbd { background:var(--card2); border:1px solid var(--border); border-radius:4px; padding:1px 5px; font-size:0.72rem; color:var(--text); }

  /* Static Summary Plot Zoom Lightbox */
  #img-lightbox { position:fixed; inset:0; background:rgba(6,8,13,0.92); z-index:1100; display:none; flex-direction:column; align-items:center; justify-content:center; padding:24px; cursor:zoom-out; }
  #img-lightbox.open { display:flex; }
  #img-lightbox img { max-width:94vw; max-height:86vh; background:#fff; border-radius:10px; padding:8px; }
  #img-lightbox .lb-cap { margin-top:10px; color:var(--text); font-size:0.9rem; }
</style>
</head>
<body>
  <h1>darkfield-hsi Workstation — __TITLE__</h1>
  <div class="kpi-grid">
    <div class="kpi"><div class="lbl">Total Blobs</div><div class="val">__N_TOTAL__</div></div>
    <div class="kpi"><div class="lbl">Single</div><div class="val" style="color:var(--green)">__N_SINGLE__</div></div>
    <div class="kpi"><div class="lbl">Likely Aggregate</div><div class="val" style="color:var(--amber)">__N_LIKELY__</div></div>
    <div class="kpi"><div class="lbl">Aggregate</div><div class="val" style="color:var(--red)">__N_AGG__</div></div>
    <div class="kpi"><div class="lbl">Tracked Pairs</div><div class="val">__N_TRACKED__</div></div>
    <div class="kpi"><div class="lbl">Median Peak</div><div class="val">__MED_PEAK__</div></div>
    <div class="kpi"><div class="lbl">Median FWHM</div><div class="val">__MED_FWHM__</div></div>
  </div>

  __QC_SECTION__
  __SUMMARY_SECTION__

  <h2>Interactive Filters &amp; Review Session</h2>
  <div class="toolbar">
    <select id="f-frame"><option value="">All Frames</option>__FRAME_OPTS__</select>
    <select id="f-class">
      <option value="">All Classes</option>
      <option value="Aggregate">Aggregate</option>
      <option value="Likely_Aggregate">Likely_Aggregate</option>
      <option value="Single">Single</option>
    </select>
    <select id="f-color"><option value="">All Colors</option>__COLOR_OPTS__</select>
    <select id="f-evidence"><option value="">All Evidence</option>__EVIDENCE_OPTS__</select>
    <select id="f-snr">
      <option value="">All SNR</option>
      <option value="good">Good SNR Only</option>
      <option value="low">Low SNR Only</option>
    </select>
    <select id="f-review">
      <option value="">All Review Status</option>
      <option value="unlabeled">Unlabeled Only</option>
      <option value="labeled">Labeled Only</option>
      <option value="Single">Reviewed: Single</option>
      <option value="Aggregate">Reviewed: Aggregate</option>
      <option value="Noise">Reviewed: Noise</option>
      <option value="disagree">Disagrees with Model</option>
    </select>
    <label class="chk"><input id="f-multipeak" type="checkbox"/> Multi-Peak (≥2)</label>
    <input id="f-min-wl" type="number" placeholder="Min Peak nm" style="width:108px"/>
    <input id="f-max-wl" type="number" placeholder="Max Peak nm" style="width:108px"/>
    <input id="f-min-fwhm" type="number" placeholder="Min FWHM nm" style="width:112px"/>
    <input id="f-max-fwhm" type="number" placeholder="Max FWHM nm" style="width:112px"/>
    <input id="f-search" type="text" placeholder="Blob ID / Evidence / Note..." style="width:175px"/>
    <select id="f-sort">
      <option value="score_desc">Sort: Aggregate Score ↓</option>
      <option value="fwhm_desc">Sort: FWHM nm ↓</option>
      <option value="peak_desc">Sort: Peak nm ↓</option>
      <option value="int_desc">Sort: Max Intensity ↓</option>
      <option value="area_desc">Sort: Spot Area ↓</option>
      <option value="id_asc">Sort: Blob ID ↑</option>
    </select>
  </div>
  <div class="toolbar" style="padding:9px 12px;">
    <button id="btn-export" class="btn">Export Labels CSV (<span id="lbl-count">0</span>)</button>
    <button id="btn-import" class="btn secondary">Import Labels CSV</button>
    <input id="file-import" type="file" accept=".csv,text/csv" style="display:none"/>
    <button id="btn-export-filtered" class="btn secondary">Export Filtered Blobs CSV (<span id="filt-count">0</span>)</button>
    <button id="btn-clear" class="btn secondary">Clear Labels</button>
    <span id="status-toast" class="toast"></span>
  </div>

  <h2>Interactive Spatial FOV Map &amp; Clickable Spectral Scatter <span class="hint">(linked to active filters — click any dot to inspect fullscreen)</span></h2>
  <div class="explore-grid">
    <div class="panel">
      <div class="panel-head">
        <strong>Dark-Field FOV Map</strong>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <select id="fov-frame-select">__FRAME_OPTS__</select>
          <select id="fov-color-by">
            <option value="class">Color by: Aggregate Class</option>
            <option value="peak">Color by: Peak Wavelength (nm)</option>
          </select>
        </div>
      </div>
      <div class="canvas-wrap"><canvas id="fov-canvas" class="interactive" width="520" height="520"></canvas></div>
    </div>
    <div class="panel">
      <div class="panel-head">
        <strong>FWHM vs. Peak Wavelength (Interactive)</strong>
        <select id="scatter-color-by">
          <option value="class">Color by: Aggregate Class</option>
          <option value="review">Color by: Human Review</option>
        </select>
      </div>
      <div class="canvas-wrap"><canvas id="scatter-canvas" class="interactive" width="560" height="520"></canvas></div>
    </div>
  </div>

  <h2>Blob Gallery <span class="hint">(click any card to open Full-Screen Inspector)</span></h2>
  <div class="pager">
    <span id="pager-info"></span>
    <div style="display:flex;gap:8px;">
      <button id="prev-page" class="btn secondary">← Prev Page</button>
      <button id="next-page" class="btn secondary">Next Page →</button>
    </div>
  </div>
  <div id="gallery" class="gallery"></div>

  <!-- Full-Screen Blob Inspector Modal -->
  <div id="inspector-modal" class="modal-backdrop" role="dialog" aria-modal="true">
    <button id="modal-prev" class="modal-nav" title="Previous filtered blob (Left Arrow)">&#8249;</button>
    <button id="modal-next" class="modal-nav" title="Next filtered blob (Right Arrow)">&#8250;</button>
    <div class="modal-box">
      <div class="modal-header">
        <div class="modal-header-left">
          <strong id="modal-title" style="font-size:1.05rem;">Blob #0</strong>
          <span id="modal-pos" class="modal-pos">1 / 1</span>
          <span id="modal-badge" class="badge cls-Single">Single</span>
        </div>
        <button id="modal-close" class="btn secondary">✕ Close (Esc)</button>
      </div>
      <div class="modal-body">
        <div class="modal-left">
          <div class="modal-spec-wrap"><img id="modal-spec-img" src="" alt="Smoothed Spectrum"/></div>
          <div id="modal-links" class="modal-links"></div>
        </div>
        <div class="modal-right">
          <div class="zoom-bar">
            <span>Crop zoom:</span>
            <button class="zoom-btn" data-zoom="1">1×</button>
            <button class="zoom-btn active" data-zoom="2">2×</button>
            <button class="zoom-btn" data-zoom="3">3×</button>
          </div>
          <div class="modal-visuals">
            <div class="crop-box"><img id="modal-norm-img" src="" alt="Normal crop"/><span>Normal</span></div>
            <div class="crop-box"><img id="modal-mark-img" src="" alt="Marked crop"/><span>Marked</span></div>
            <div class="minimap-box"><canvas id="modal-fov-canvas" width="156" height="156"></canvas><span id="modal-xy-lbl">FOV (X, Y)</span></div>
          </div>
          <div id="modal-evidence"></div>
          <div id="modal-details" class="detail-grid"></div>
          <div class="modal-review">
            <div class="review-row">
              <button class="review-btn" id="m-btn-Single" data-mval="Single">1 · Single</button>
              <button class="review-btn" id="m-btn-Aggregate" data-mval="Aggregate">2 · Aggregate</button>
              <button class="review-btn" id="m-btn-Noise" data-mval="Noise">3 · Noise</button>
            </div>
            <input id="modal-note" type="text" placeholder="Optional note for this blob (saved automatically)..."/>
            <div class="kbd-hint">
              <label style="display:inline-flex;align-items:center;gap:5px;cursor:pointer;">
                <input id="auto-advance" type="checkbox"/> Auto-advance after labeling
              </label>
              <span><kbd>←</kbd>/<kbd>→</kbd> Navigate · <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> Label · <kbd>Esc</kbd> Close</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Static Summary Plot Zoom Lightbox -->
  <div id="img-lightbox">
    <img id="lb-img" src="" alt="Zoomed summary chart"/>
    <div id="lb-cap" class="lb-cap"></div>
  </div>

  <div id="canvas-tip" class="canvas-tip"></div>

  <script id="blob-data" type="application/json">__BLOB_JSON__</script>
  <script id="fov-data" type="application/json">__FOV_JSON__</script>
  <script>
  (function() {
    const DATA = JSON.parse(document.getElementById("blob-data").textContent);
    const FOV_META = JSON.parse(document.getElementById("fov-data").textContent);
    const PAGE_SIZE = __PAGE_SIZE__;
    const LABEL_KEY = "darkfield_hsi_labels_" + location.pathname;
    const NOTE_KEY = "darkfield_hsi_notes_" + location.pathname;
    const PREF_KEY = "darkfield_hsi_prefs_" + location.pathname;

    let labels = JSON.parse(localStorage.getItem(LABEL_KEY) || "{}");
    let notes = JSON.parse(localStorage.getItem(NOTE_KEY) || "{}");
    let prefs = JSON.parse(localStorage.getItem(PREF_KEY) || '{"zoom":2,"autoAdvance":false}');

    let page = 0;
    let filtered = DATA.slice();
    let modalIdx = -1;
    let fovImages = {};
    let fovDrawPts = [];
    let scatterDrawPts = [];

    const el = (id) => document.getElementById(id);
    const itemKey = (d) => d.file + "::" + d.frame + "::" + d.id;
    const CLS_COLOR = { Single: "#2ecc71", Likely_Aggregate: "#f39c12", Aggregate: "#e74c3c" };
    const REV_COLOR = { Single: "#2ecc71", Aggregate: "#e74c3c", Noise: "#f39c12", "": "#5b6780" };

    function showToast(msg) {
      const t = el("status-toast");
      t.textContent = msg;
      clearTimeout(t._timer);
      t._timer = setTimeout(() => { t.textContent = ""; }, 3500);
    }

    function saveStorage() {
      localStorage.setItem(LABEL_KEY, JSON.stringify(labels));
      localStorage.setItem(NOTE_KEY, JSON.stringify(notes));
      localStorage.setItem(PREF_KEY, JSON.stringify(prefs));
      updateCounts();
    }

    function updateCounts() {
      el("lbl-count").textContent = Object.keys(labels).length;
      el("filt-count").textContent = filtered.length;
    }

    function wavelengthToColor(nm) {
      if (nm === null || nm === undefined) return "#94a3b8";
      const t = Math.max(0, Math.min(1, (nm - 440) / 260));
      const hue = 240 * (1 - t); // 440nm=blue -> 540nm=green -> 700nm=red
      return `hsl(${hue.toFixed(0)}, 88%, 55%)`;
    }

    function ensureFovImage(stem, cb) {
      const meta = FOV_META[stem];
      if (!meta || !meta.src) { cb(null, null); return; }
      if (fovImages[stem] && fovImages[stem].complete) { cb(fovImages[stem], meta); return; }
      const img = fovImages[stem] || new Image();
      fovImages[stem] = img;
      img.onload = () => cb(img, meta);
      img.onerror = () => cb(null, meta);
      if (!img.src) img.src = meta.src;
    }

    function disagreesWithModel(d, lbl) {
      if (!lbl) return false;
      const modelAgg = d.agg_class === "Aggregate";
      if (lbl === "Aggregate" && !modelAgg) return true;
      if (lbl === "Single" && modelAgg) return true;
      if (lbl === "Noise") return true;
      return false;
    }

    function applyFilters() {
      const fFrame = el("f-frame").value;
      const fClass = el("f-class").value;
      const fColor = el("f-color").value;
      const fEv = el("f-evidence").value;
      const fSnr = el("f-snr").value;
      const fRev = el("f-review").value;
      const fMulti = el("f-multipeak").checked;
      const minWl = el("f-min-wl").value === "" ? null : parseFloat(el("f-min-wl").value);
      const maxWl = el("f-max-wl").value === "" ? null : parseFloat(el("f-max-wl").value);
      const minFw = el("f-min-fwhm").value === "" ? null : parseFloat(el("f-min-fwhm").value);
      const maxFw = el("f-max-fwhm").value === "" ? null : parseFloat(el("f-max-fwhm").value);
      const q = el("f-search").value.trim().toLowerCase();
      const sortBy = el("f-sort").value;

      if (fFrame && el("fov-frame-select").value !== fFrame) {
        el("fov-frame-select").value = fFrame;
      }

      filtered = DATA.filter((d) => {
        const key = itemKey(d);
        const lbl = labels[key] || "";
        const note = (notes[key] || "").toLowerCase();
        if (fFrame && d.file !== fFrame) return false;
        if (fClass && d.agg_class !== fClass) return false;
        if (fColor && d.color !== fColor) return false;
        if (fEv && !d.evidence.split("+").includes(fEv)) return false;
        if (fSnr === "good" && d.low_snr) return false;
        if (fSnr === "low" && !d.low_snr) return false;
        if (fMulti && !(d.num_peaks !== null && d.num_peaks >= 2)) return false;
        if (fRev === "unlabeled" && lbl) return false;
        if (fRev === "labeled" && !lbl) return false;
        if ((fRev === "Single" || fRev === "Aggregate" || fRev === "Noise") && lbl !== fRev) return false;
        if (fRev === "disagree" && !disagreesWithModel(d, lbl)) return false;
        if (minWl !== null && (d.peak_nm === null || d.peak_nm < minWl)) return false;
        if (maxWl !== null && (d.peak_nm === null || d.peak_nm > maxWl)) return false;
        if (minFw !== null && (d.fwhm_nm === null || d.fwhm_nm < minFw)) return false;
        if (maxFw !== null && (d.fwhm_nm === null || d.fwhm_nm > maxFw)) return false;
        if (q) {
          const hay = (String(d.id) + " " + d.evidence + " " + d.color + " " + note).toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

      filtered.sort((a, b) => {
        if (sortBy === "score_desc") return (b.agg_score ?? -1) - (a.agg_score ?? -1);
        if (sortBy === "fwhm_desc") return (b.fwhm_nm ?? -1) - (a.fwhm_nm ?? -1);
        if (sortBy === "peak_desc") return (b.peak_nm ?? -1) - (a.peak_nm ?? -1);
        if (sortBy === "int_desc") return (b.max_int ?? -1) - (a.max_int ?? -1);
        if (sortBy === "area_desc") return (b.area ?? -1) - (a.area ?? -1);
        return a.id - b.id;
      });

      page = 0;
      updateCounts();
      renderPage();
      drawFovMap();
      drawScatter();
      if (modalIdx >= 0) {
        if (filtered.length === 0) closeModal();
        else {
          modalIdx = Math.min(modalIdx, filtered.length - 1);
          renderModal();
        }
      }
    }

    function setLabel(key, val) {
      if (labels[key] === val) delete labels[key];
      else labels[key] = val;
      saveStorage();
      renderPage();
      drawFovMap();
      drawScatter();
    }

    function renderPage() {
      const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
      page = Math.max(0, Math.min(page, totalPages - 1));
      const startIdx = page * PAGE_SIZE;
      const slice = filtered.slice(startIdx, startIdx + PAGE_SIZE);
      el("pager-info").textContent = `Showing ${slice.length} of ${filtered.length} matching blobs (Page ${page + 1} / ${totalPages})`;

      const root = el("gallery");
      root.innerHTML = "";
      slice.forEach((d, offset) => {
        const globalIdx = startIdx + offset;
        const key = itemKey(d);
        const cur = labels[key] || "";
        const card = document.createElement("div");
        card.className = "blob-card";
        card.innerHTML = `
          <div class="blob-head">
            <strong>Blob #${d.id} <span style="color:var(--muted);font-weight:400">(F${d.frame})</span></strong>
            <span class="badge cls-${d.agg_class}">${d.agg_class} (${d.agg_score})</span>
          </div>
          <div class="crops-row">
            ${d.norm ? `<img class="crop" src="${d.norm}" title="Normal Crop" loading="lazy"/>` : ""}
            ${d.mark ? `<img class="crop" src="${d.mark}" title="Marked Crop" loading="lazy"/>` : ""}
            <div class="meta" style="flex:1">
              <span>Color: <b>${d.color}</b></span>
              <span>Area: <b>${d.area ?? "-"} px</b></span>
              <span>Peak: <b>${d.peak_nm !== null ? d.peak_nm + " nm" : "-"}</b></span>
              <span>FWHM: <b>${d.fwhm_nm !== null ? d.fwhm_nm + " nm" : "-"}</b></span>
              <span>SNR: <b>${d.snr !== null ? d.snr : "-"}</b></span>
              <span>Ecc: <b>${d.ecc !== null ? d.ecc : "-"}</b></span>
              <span style="grid-column:1/-1">Evidence: <b>${d.evidence}</b></span>
            </div>
          </div>
          ${d.spec ? `<img class="spec-img" src="${d.spec}" loading="lazy"/>` : ""}
          <div class="review-row">
            <button class="review-btn ${cur === 'Single' ? 'active-Single' : ''}" data-val="Single">Single</button>
            <button class="review-btn ${cur === 'Aggregate' ? 'active-Aggregate' : ''}" data-val="Aggregate">Aggregate</button>
            <button class="review-btn ${cur === 'Noise' ? 'active-Noise' : ''}" data-val="Noise">Noise</button>
          </div>
        `;
        card.addEventListener("click", (e) => {
          if (e.target.closest(".review-btn")) return;
          openModal(globalIdx);
        });
        card.querySelectorAll(".review-btn").forEach((btn) => {
          btn.addEventListener("click", (e) => {
            e.stopPropagation();
            setLabel(key, btn.dataset.val);
          });
        });
        root.appendChild(card);
      });
    }

    // ---------- Interactive FOV Map ----------
    function drawFovMap(hoverIdx = -1) {
      const canvas = el("fov-canvas");
      const ctx = canvas.getContext("2d");
      const stem = el("fov-frame-select").value;
      const colorMode = el("fov-color-by").value;
      const W = canvas.width, H = canvas.height;

      ensureFovImage(stem, (img, meta) => {
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#080a0f";
        ctx.fillRect(0, 0, W, H);

        const imgW = meta ? meta.width : 696;
        const imgH = meta ? meta.height : 696;
        const scale = Math.min(W / imgW, H / imgH);
        const drawW = imgW * scale, drawH = imgH * scale;
        const offX = (W - drawW) / 2, offY = (H - drawH) / 2;

        if (img) ctx.drawImage(img, offX, offY, drawW, drawH);

        fovDrawPts = [];
        filtered.forEach((d, idx) => {
          if (d.file !== stem) return;
          const px = offX + d.x * scale;
          const py = offY + d.y * scale;
          const fill = colorMode === "peak" ? wavelengthToColor(d.peak_nm) : (CLS_COLOR[d.agg_class] || "#4ea8de");
          const isHov = idx === hoverIdx;
          const r = isHov ? 7 : (d.agg_class === "Aggregate" ? 4.8 : 3.4);

          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fillStyle = fill;
          ctx.globalAlpha = isHov ? 1.0 : 0.82;
          ctx.fill();
          ctx.globalAlpha = 1.0;
          ctx.lineWidth = isHov ? 2.2 : (labels[itemKey(d)] ? 1.6 : 0.8);
          ctx.strokeStyle = isHov ? "#ffffff" : (labels[itemKey(d)] ? "#ffffff" : "#000000");
          ctx.stroke();

          fovDrawPts.push({ px, py, idx, d });
        });
      });
    }

    // ---------- Interactive FWHM vs Peak Scatter ----------
    function drawScatter(hoverIdx = -1) {
      const canvas = el("scatter-canvas");
      const ctx = canvas.getContext("2d");
      const colorMode = el("scatter-color-by").value;
      const W = canvas.width, H = canvas.height;
      const pad = { l: 54, r: 20, t: 20, b: 44 };
      const plotW = W - pad.l - pad.r;
      const plotH = H - pad.t - pad.b;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#0b0e16";
      ctx.fillRect(0, 0, W, H);

      const pts = [];
      filtered.forEach((d, idx) => {
        if (d.peak_nm !== null && d.fwhm_nm !== null) pts.push({ idx, d });
      });
      scatterDrawPts = [];

      if (pts.length === 0) {
        ctx.fillStyle = "#95a0b5";
        ctx.font = "14px system-ui";
        ctx.textAlign = "center";
        ctx.fillText("No spectral Peak/FWHM data for current filters", W / 2, H / 2);
        return;
      }

      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of pts) {
        if (p.d.peak_nm < minX) minX = p.d.peak_nm;
        if (p.d.peak_nm > maxX) maxX = p.d.peak_nm;
        if (p.d.fwhm_nm < minY) minY = p.d.fwhm_nm;
        if (p.d.fwhm_nm > maxY) maxY = p.d.fwhm_nm;
      }
      const spanX = Math.max(20, maxX - minX);
      const spanY = Math.max(20, maxY - minY);
      minX -= spanX * 0.06; maxX += spanX * 0.06;
      minY = Math.max(0, minY - spanY * 0.06); maxY += spanY * 0.08;

      const toX = (v) => pad.l + ((v - minX) / (maxX - minX)) * plotW;
      const toY = (v) => pad.t + plotH - ((v - minY) / (maxY - minY)) * plotH;

      // Grid & ticks
      ctx.strokeStyle = "rgba(148,163,184,0.16)";
      ctx.lineWidth = 1;
      ctx.fillStyle = "#95a0b5";
      ctx.font = "11px system-ui";
      for (let i = 0; i <= 5; i++) {
        const vx = minX + (i / 5) * (maxX - minX);
        const gx = toX(vx);
        ctx.beginPath(); ctx.moveTo(gx, pad.t); ctx.lineTo(gx, pad.t + plotH); ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillText(vx.toFixed(0), gx, pad.t + plotH + 16);

        const vy = minY + (i / 5) * (maxY - minY);
        const gy = toY(vy);
        ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(pad.l + plotW, gy); ctx.stroke();
        ctx.textAlign = "right";
        ctx.fillText(vy.toFixed(0), pad.l - 7, gy + 4);
      }

      // Axis borders & titles
      ctx.strokeStyle = "#3b455e";
      ctx.strokeRect(pad.l, pad.t, plotW, plotH);
      ctx.fillStyle = "#cbd5e1";
      ctx.font = "12px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("Peak Wavelength (nm)", pad.l + plotW / 2, H - 10);
      ctx.save();
      ctx.translate(15, pad.t + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("FWHM (nm)", 0, 0);
      ctx.restore();

      for (const p of pts) {
        const px = toX(p.d.peak_nm);
        const py = toY(p.d.fwhm_nm);
        const isHov = p.idx === hoverIdx;
        const rev = labels[itemKey(p.d)] || "";
        const fill = colorMode === "review" ? (REV_COLOR[rev] || "#5b6780") : (CLS_COLOR[p.d.agg_class] || "#4ea8de");

        ctx.beginPath();
        ctx.arc(px, py, isHov ? 6.5 : 3.6, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.globalAlpha = isHov ? 1.0 : 0.72;
        ctx.fill();
        ctx.globalAlpha = 1.0;
        if (isHov) {
          ctx.lineWidth = 2;
          ctx.strokeStyle = "#ffffff";
          ctx.stroke();
        }
        scatterDrawPts.push({ px, py, idx: p.idx, d: p.d });
      }
    }

    function bindCanvasHoverClick(canvasId, getPts, redraw) {
      const canvas = el(canvasId);
      const tip = el("canvas-tip");
      let lastHov = -1;

      function hitTest(e) {
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
        const my = (e.clientY - rect.top) * (canvas.height / rect.height);
        let best = null, bestD2 = 14 * 14;
        for (const p of getPts()) {
          const d2 = (p.px - mx) ** 2 + (p.py - my) ** 2;
          if (d2 < bestD2) { bestD2 = d2; best = p; }
        }
        return best;
      }

      canvas.addEventListener("mousemove", (e) => {
        const hit = hitTest(e);
        const nextHov = hit ? hit.idx : -1;
        if (nextHov !== lastHov) { lastHov = nextHov; redraw(lastHov); }
        if (hit) {
          const d = hit.d;
          const rev = labels[itemKey(d)];
          tip.innerHTML = `<strong>Blob #${d.id}</strong> (F${d.frame})<br/>Class: <b>${d.agg_class}</b> (${d.agg_score})<br/>Peak: <b>${d.peak_nm ?? "-"} nm</b> · FWHM: <b>${d.fwhm_nm ?? "-"} nm</b><br/>Evidence: ${d.evidence}${rev ? `<br/>Reviewed: <b>${rev}</b>` : ""}`;
          tip.style.display = "block";
          tip.style.left = (e.clientX + 14) + "px";
          tip.style.top = (e.clientY + 14) + "px";
        } else {
          tip.style.display = "none";
        }
      });
      canvas.addEventListener("mouseleave", () => {
        tip.style.display = "none";
        if (lastHov !== -1) { lastHov = -1; redraw(-1); }
      });
      canvas.addEventListener("click", (e) => {
        const hit = hitTest(e);
        if (hit) { tip.style.display = "none"; openModal(hit.idx); }
      });
    }

    // ---------- Full-Screen Blob Inspector Modal ----------
    function applyCropZoom() {
      const z = prefs.zoom || 2;
      const px = z === 1 ? 72 : (z === 2 ? 128 : 180);
      el("modal-norm-img").style.width = px + "px";
      el("modal-norm-img").style.height = px + "px";
      el("modal-mark-img").style.width = px + "px";
      el("modal-mark-img").style.height = px + "px";
      document.querySelectorAll(".zoom-btn").forEach((b) => {
        b.classList.toggle("active", parseInt(b.dataset.zoom, 10) === z);
      });
    }

    function drawModalMinimap(d) {
      const canvas = el("modal-fov-canvas");
      const ctx = canvas.getContext("2d");
      const W = canvas.width, H = canvas.height;
      el("modal-xy-lbl").textContent = `FOV (${d.x}, ${d.y})`;

      ensureFovImage(d.file, (img, meta) => {
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "#000";
        ctx.fillRect(0, 0, W, H);
        const imgW = meta ? meta.width : 696;
        const imgH = meta ? meta.height : 696;
        if (img) ctx.drawImage(img, 0, 0, W, H);

        const cx = (d.x / imgW) * W;
        const cy = (d.y / imgH) * H;
        ctx.strokeStyle = "rgba(78, 168, 222, 0.85)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
        ctx.strokeStyle = "#ff3b30";
        ctx.lineWidth = 2;
        ctx.strokeRect(cx - 7, cy - 7, 14, 14);
      });
    }

    function openModal(idx) {
      if (idx < 0 || idx >= filtered.length) return;
      modalIdx = idx;
      el("inspector-modal").classList.add("open");
      document.body.style.overflow = "hidden";
      renderModal();
    }

    function closeModal() {
      modalIdx = -1;
      el("inspector-modal").classList.remove("open");
      document.body.style.overflow = "";
    }

    function stepModal(delta) {
      if (modalIdx < 0) return;
      const next = modalIdx + delta;
      if (next >= 0 && next < filtered.length) {
        modalIdx = next;
        renderModal();
      }
    }

    function renderModal() {
      if (modalIdx < 0 || modalIdx >= filtered.length) return;
      const d = filtered[modalIdx];
      const key = itemKey(d);
      const curLbl = labels[key] || "";

      el("modal-title").textContent = `Blob #${d.id} — ${d.file} (Frame ${d.frame})`;
      el("modal-pos").textContent = `${modalIdx + 1} / ${filtered.length}`;
      const badge = el("modal-badge");
      badge.className = `badge cls-${d.agg_class}`;
      badge.textContent = `${d.agg_class} (score ${d.agg_score})`;

      el("modal-prev").disabled = modalIdx <= 0;
      el("modal-next").disabled = modalIdx >= filtered.length - 1;

      const specImg = el("modal-spec-img");
      if (d.spec) { specImg.src = d.spec; specImg.parentElement.style.display = "block"; }
      else { specImg.parentElement.style.display = "none"; }

      let linksHtml = "";
      if (d.spec_csv) linksHtml += `<a href="${d.spec_csv}" download>⬇ Download Spectrum CSV</a>`;
      if (d.spec) linksHtml += `<a href="${d.spec}" target="_blank">↗ Open Spectrum PNG</a>`;
      el("modal-links").innerHTML = linksHtml;

      el("modal-norm-img").src = d.norm || "";
      el("modal-mark-img").src = d.mark || "";
      applyCropZoom();
      drawModalMinimap(d);

      const tags = d.evidence.split("+").map((t) => `<span class="ev-chip">${t}</span>`).join("");
      el("modal-evidence").innerHTML = `<span style="font-size:0.75rem;color:var(--muted);margin-right:6px;">Evidence:</span>${tags}`;

      const rows = [
        ["Peak Wavelength", d.peak_nm !== null ? `${d.peak_nm} nm (B${d.peak_band})` : "N/A"],
        ["FWHM", d.fwhm_nm !== null ? `${d.fwhm_nm} nm` : "N/A"],
        ["Max Intensity", d.max_int ?? "N/A"],
        ["Spectral SNR", d.snr !== null ? `${d.snr}${d.low_snr ? " (Low)" : ""}` : "N/A"],
        ["LSPR Peaks", d.num_peaks !== null ? `${d.num_peaks}${d.sec_peak_nm !== null ? " (2nd: " + d.sec_peak_nm + " nm)" : ""}` : "N/A"],
        ["Spot Area / R", `${d.area ?? "-"} px / ${d.spot_r ?? "-"} px`],
        ["Eccentricity", d.ecc ?? "-"],
        ["Spot Peak", `${d.spot_peak ?? "-"}${d.sat_px ? " (Sat)" : ""}`],
        ["Color", `${d.color} (${d.fg_px ?? "-"} fg px)`],
        ["Hue / Sat / Val", `${d.hue ?? "-"}° / ${d.sat ?? "-"} / ${d.val ?? "-"}`],
      ];
      el("modal-details").innerHTML = rows.map(([k, v]) => `<div class="detail-item"><span>${k}</span><b>${v}</b></div>`).join("");

      ["Single", "Aggregate", "Noise"].forEach((val) => {
        const b = el("m-btn-" + val);
        b.className = "review-btn" + (curLbl === val ? ` active-${val}` : "");
      });
      el("modal-note").value = notes[key] || "";
    }

    function setLabelFromModal(val) {
      if (modalIdx < 0 || modalIdx >= filtered.length) return;
      const d = filtered[modalIdx];
      const key = itemKey(d);
      const wasSame = labels[key] === val;
      setLabel(key, val);
      if (!wasSame && el("auto-advance").checked && modalIdx < filtered.length - 1) {
        modalIdx++;
      }
      renderModal();
    }

    // ---------- CSV Helpers (RFC-4180 Export & Import) ----------
    function csvCell(v) {
      const s = v === null || v === undefined ? "" : String(v);
      return /[",\\r\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    }

    function parseCsvLine(line) {
      const out = [];
      let cur = "", inQ = false;
      for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (inQ) {
          if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
          else if (ch === '"') inQ = false;
          else cur += ch;
        } else {
          if (ch === '"') inQ = true;
          else if (ch === ",") { out.push(cur); cur = ""; }
          else cur += ch;
        }
      }
      out.push(cur);
      return out;
    }

    function downloadCsv(filename, lines) {
      const blob = new Blob([lines.join("\\n") + "\\n"], { type: "text/csv;charset=utf-8;" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
    }

    // ---------- Event Bindings ----------
    ["f-frame","f-class","f-color","f-evidence","f-snr","f-review","f-multipeak","f-min-wl","f-max-wl","f-min-fwhm","f-max-fwhm","f-search","f-sort"].forEach((id) => {
      el(id).addEventListener("input", applyFilters);
      el(id).addEventListener("change", applyFilters);
    });
    el("fov-frame-select").addEventListener("change", () => drawFovMap());
    el("fov-color-by").addEventListener("change", () => drawFovMap());
    el("scatter-color-by").addEventListener("change", () => drawScatter());

    bindCanvasHoverClick("fov-canvas", () => fovDrawPts, drawFovMap);
    bindCanvasHoverClick("scatter-canvas", () => scatterDrawPts, drawScatter);

    el("prev-page").addEventListener("click", () => { if (page > 0) { page--; renderPage(); } });
    el("next-page").addEventListener("click", () => { page++; renderPage(); });

    el("modal-prev").addEventListener("click", () => stepModal(-1));
    el("modal-next").addEventListener("click", () => stepModal(1));
    el("modal-close").addEventListener("click", closeModal);
    el("inspector-modal").addEventListener("click", (e) => {
      if (e.target === el("inspector-modal")) closeModal();
    });

    ["Single", "Aggregate", "Noise"].forEach((val) => {
      el("m-btn-" + val).addEventListener("click", () => setLabelFromModal(val));
    });

    el("modal-note").addEventListener("input", (e) => {
      if (modalIdx < 0 || modalIdx >= filtered.length) return;
      const key = itemKey(filtered[modalIdx]);
      const val = e.target.value;
      if (val.trim()) notes[key] = val;
      else delete notes[key];
      saveStorage();
    });

    el("auto-advance").checked = !!prefs.autoAdvance;
    el("auto-advance").addEventListener("change", (e) => {
      prefs.autoAdvance = e.target.checked;
      saveStorage();
    });

    document.querySelectorAll(".zoom-btn").forEach((b) => {
      b.addEventListener("click", () => {
        prefs.zoom = parseInt(b.dataset.zoom, 10);
        saveStorage();
        applyCropZoom();
      });
    });

    // Static summary plot zoom lightbox
    document.querySelectorAll(".plot-card").forEach((card) => {
      card.addEventListener("click", () => {
        el("lb-img").src = card.dataset.zoomSrc;
        el("lb-cap").textContent = card.dataset.zoomTitle || "";
        el("img-lightbox").classList.add("open");
      });
    });
    el("img-lightbox").addEventListener("click", () => el("img-lightbox").classList.remove("open"));

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) => {
      if (el("img-lightbox").classList.contains("open")) {
        if (e.key === "Escape") el("img-lightbox").classList.remove("open");
        return;
      }
      if (modalIdx < 0) return;

      const tag = (document.activeElement && document.activeElement.tagName) || "";
      const typing = tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
      if (e.key === "Escape") {
        if (typing) document.activeElement.blur();
        else closeModal();
        return;
      }
      if (typing) return;

      if (e.key === "ArrowLeft") { e.preventDefault(); stepModal(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); stepModal(1); }
      else if (e.key === "1") { e.preventDefault(); setLabelFromModal("Single"); }
      else if (e.key === "2") { e.preventDefault(); setLabelFromModal("Aggregate"); }
      else if (e.key === "3") { e.preventDefault(); setLabelFromModal("Noise"); }
    });

    // Export Labels CSV
    el("btn-export").addEventListener("click", () => {
      const header = "Filename,Frame_Number,Blob_ID,Human_Label,Note,Predicted_Class,Aggregate_Score,Aggregate_Evidence,Peak_nm,FWHM_nm";
      const rows = [header];
      for (const d of DATA) {
        const key = itemKey(d);
        const lbl = labels[key] || "";
        const note = notes[key] || "";
        if (!lbl && !note) continue;
        rows.push([d.file, d.frame, d.id, lbl, note, d.agg_class, d.agg_score, d.evidence, d.peak_nm ?? "", d.fwhm_nm ?? ""].map(csvCell).join(","));
      }
      downloadCsv("human_review_labels.csv", rows);
      showToast(`Exported ${rows.length - 1} labeled/noted row(s).`);
    });

    // Import Labels CSV
    el("btn-import").addEventListener("click", () => el("file-import").click());
    el("file-import").addEventListener("change", (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const lines = String(reader.result || "").split(/\\r?\\n/).filter((l) => l.trim());
        if (lines.length < 2) return;
        const hdr = parseCsvLine(lines[0]).map((s) => s.trim());
        const iFile = hdr.indexOf("Filename");
        const iFrame = hdr.indexOf("Frame_Number");
        const iId = hdr.indexOf("Blob_ID");
        const iLbl = hdr.indexOf("Human_Label");
        const iNote = hdr.indexOf("Note");
        if (iFile < 0 || iFrame < 0 || iId < 0) {
          showToast("Invalid CSV: missing Filename/Frame_Number/Blob_ID.");
          return;
        }
        let imported = 0;
        for (let i = 1; i < lines.length; i++) {
          const cols = parseCsvLine(lines[i]);
          const key = cols[iFile] + "::" + cols[iFrame] + "::" + cols[iId];
          const lbl = iLbl >= 0 ? (cols[iLbl] || "").trim() : "";
          const nt = iNote >= 0 ? (cols[iNote] || "").trim() : "";
          if (lbl) { labels[key] = lbl; imported++; }
          if (nt) notes[key] = nt;
        }
        saveStorage();
        applyFilters();
        showToast(`Imported ${imported} label(s) from ${file.name}.`);
      };
      reader.readAsText(file);
      e.target.value = "";
    });

    // Export Filtered Blobs CSV
    el("btn-export-filtered").addEventListener("click", () => {
      const header = "Filename,Frame_Number,Blob_ID,X,Y,Color,Aggregate_Class,Aggregate_Score,Aggregate_Evidence,Peak_nm,FWHM_nm,Max_Intensity,Spectral_SNR,Num_Peaks,Spot_Area_px,Eccentricity,Human_Label,Note";
      const rows = [header];
      for (const d of filtered) {
        const key = itemKey(d);
        rows.push([
          d.file, d.frame, d.id, d.x, d.y, d.color, d.agg_class, d.agg_score, d.evidence,
          d.peak_nm ?? "", d.fwhm_nm ?? "", d.max_int ?? "", d.snr ?? "", d.num_peaks ?? "",
          d.area ?? "", d.ecc ?? "", labels[key] || "", notes[key] || ""
        ].map(csvCell).join(","));
      }
      downloadCsv("filtered_blobs.csv", rows);
      showToast(`Exported ${filtered.length} filtered blob(s).`);
    });

    el("btn-clear").addEventListener("click", () => {
      if (!confirm("Clear all saved review labels and notes for this report?")) return;
      labels = {};
      notes = {};
      saveStorage();
      applyFilters();
      showToast("Cleared all saved labels & notes.");
    });

    applyCropZoom();
    updateCounts();
    applyFilters();
  })();
  </script>
</body>
</html>
"""
