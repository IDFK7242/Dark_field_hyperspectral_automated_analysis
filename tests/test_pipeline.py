"""Unit, synthetic CI end-to-end, and optional real-CytoViva integration tests."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from darkfield_hsi.aggregate import classify_aggregates
from darkfield_hsi.cli import main as cli_main
from darkfield_hsi.config import (
    AggregateConfig,
    PipelineConfig,
    SpectraConfig,
    load_config,
    save_config,
)
from darkfield_hsi.detection import estimate_drift, prepare_detection_image, track_blobs
from darkfield_hsi.envi import EnviCube, parse_envi_header
from darkfield_hsi.io_utils import FrameInput, discover_inputs
from darkfield_hsi.pipeline import run_batch, run_pipeline
from darkfield_hsi.postprocess import coarse_color_name
from darkfield_hsi.qc import run_frame_qc
from darkfield_hsi.spectra import analyze_spectrum

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = REPO_ROOT / "test_data"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


def _create_tiny_synthetic_dataset(root: Path) -> Path:
    """Create a self-contained 2-frame dark-field TIFF + CytoViva-style ENVI dataset for CI."""
    tiff_dir = root / "tiff"
    envi_dir = root / "envi"
    tiff_dir.mkdir(parents=True, exist_ok=True)
    envi_dir.mkdir(parents=True, exist_ok=True)

    height, width = 128, 128
    bands = 61
    wavelengths = np.linspace(400.0, 760.0, bands)
    yy, xx = np.mgrid[0:height, 0:width]

    # Known stage drift between frame 0 and frame 1: particles move by (dy=-1, dx=+2)
    frame_specs = [
        ("sample_t0", 0, 0),
        ("sample_t5min", -1, 2),
    ]

    rng = np.random.default_rng(1234)
    wl_str = ", ".join(f"{w:.4f}" for w in wavelengths)

    for stem, dy_shift, dx_shift in frame_specs:
        # Realistic dark-field background + subtle column striping
        bg = rng.normal(12.0, 2.0, size=(height, width, 3))
        col_stripe = rng.normal(0.0, 1.5, size=(1, width, 1))
        img = bg + col_stripe

        # BIL ENVI cube: (lines, bands, samples), int16
        cube_bil = rng.normal(5.0, 1.5, size=(height, bands, width)).astype(np.float64)

        # Particle 0: Green monomer AgNP (narrow single peak @ 520 nm)
        cx0, cy0 = 36 + dx_shift, 42 + dy_shift
        g0 = np.exp(-((xx - cx0) ** 2 + (yy - cy0) ** 2) / (2.0 * 1.6**2))
        img[..., 0] += 20.0 * g0
        img[..., 1] += 210.0 * g0
        img[..., 2] += 30.0 * g0
        spec0 = 180.0 * np.exp(-0.5 * ((wavelengths - 520.0) / 22.0) ** 2)
        cube_bil += g0[:, None, :] * spec0[None, :, None]

        # Particle 1: Bright, elongated, broad red-shifted bimodal aggregate (540 + 660 nm)
        cx1, cy1 = 88 + dx_shift, 82 + dy_shift
        g1 = np.exp(-(((xx - cx1) / 3.8) ** 2 + ((yy - cy1) / 1.7) ** 2) / 2.0)
        img[..., 0] += 245.0 * g1
        img[..., 1] += 235.0 * g1
        img[..., 2] += 70.0 * g1
        spec1 = (
            420.0 * np.exp(-0.5 * ((wavelengths - 540.0) / 38.0) ** 2)
            + 680.0 * np.exp(-0.5 * ((wavelengths - 660.0) / 55.0) ** 2)
        )
        cube_bil += g1[:, None, :] * spec1[None, :, None]

        # Particle 2: Second crisp green monomer (505 nm)
        cx2, cy2 = 96 + dx_shift, 30 + dy_shift
        g2 = np.exp(-((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2.0 * 1.5**2))
        img[..., 0] += 15.0 * g2
        img[..., 1] += 195.0 * g2
        img[..., 2] += 45.0 * g2
        spec2 = 165.0 * np.exp(-0.5 * ((wavelengths - 505.0) / 20.0) ** 2)
        cube_bil += g2[:, None, :] * spec2[None, :, None]

        img_u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        Image.fromarray(img_u8, mode="RGB").save(tiff_dir / f"{stem}.tif")

        # Write ENVI BIL binary + CytoViva-style multi-line .hdr
        np.rint(cube_bil).astype("<i2").tofile(envi_dir / stem)
        hdr_text = (
            "ENVI\n"
            "description = {\n"
            "CytoViva Synthetic Test Image,\n"
            "Date = Thu Nov 26 10:00:00 2026,\n"
            "Exposure time = 0.200000\n"
            "}\n"
            f"samples = {width}\n"
            f"lines = {height}\n"
            f"bands = {bands}\n"
            "header offset = 0\n"
            "file type = ENVI Standard\n"
            "data type = 2\n"
            "interleave = bil\n"
            "byte order = 0\n"
            f"wavelength = {{\n{wl_str}\n}}\n"
        )
        (envi_dir / f"{stem}.hdr").write_text(hdr_text, encoding="utf-8")

    return root


class TestConfigV02(unittest.TestCase):
    def test_load_default_yaml_and_roundtrip(self) -> None:
        cfg = load_config(DEFAULT_CONFIG_PATH)
        self.assertTrue(cfg.qc.enabled)
        self.assertTrue(cfg.detection.clean_enabled)
        self.assertEqual(cfg.detection.channel_projection, "max_rgb")
        self.assertEqual(cfg.spectra.smoothing_window, 21)
        self.assertEqual(cfg.spectra.smoothing_poly, 3)
        self.assertTrue(cfg.spectra.save_plots)
        self.assertTrue(cfg.spectra.save_per_blob_csv)
        self.assertEqual(cfg.aggregate.material_preset, "agnp")
        self.assertEqual(cfg.aggregate.resolved_red_shift_nm(), 580.0)
        self.assertEqual(cfg.aggregate.resolved_broad_fwhm_nm(), 145.0)

        with tempfile.TemporaryDirectory() as tmp:
            out_yaml = Path(tmp) / "roundtrip.yaml"
            save_config(cfg, out_yaml)
            reloaded = load_config(out_yaml)
            self.assertEqual(cfg.to_dict(), reloaded.to_dict())

    def test_validation_rejects_invalid_params(self) -> None:
        cfg = PipelineConfig()
        cfg.spectra.smoothing_window = 12  # even
        with self.assertRaises(ValueError):
            cfg.validate()

        cfg2 = PipelineConfig()
        cfg2.aggregate.material_preset = "unobtanium"
        with self.assertRaises(ValueError):
            cfg2.validate()

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text("detection:\n  nonexistent_key: 42\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(bad)


class TestCoreAlgorithmUnits(unittest.TestCase):
    def test_drift_aligns_shifted_blobs_and_preserves_true_coords(self) -> None:
        ref = np.zeros((128, 128), dtype=np.float32)
        mov = np.zeros((128, 128), dtype=np.float32)
        pts_ref = [(30, 40), (80, 75), (50, 95)]
        dy_obj, dx_obj = 3, -2
        for y, x in pts_ref:
            cv2.circle(ref, (x, y), 2, 1.0, -1)
            cv2.circle(mov, (x + dx_obj, y + dy_obj), 2, 1.0, -1)

        sy, sx, _ = estimate_drift(ref, mov, upsample_factor=10)
        self.assertAlmostEqual(sy, -dy_obj, delta=0.25)
        self.assertAlmostEqual(sx, -dx_obj, delta=0.25)

        prev_blobs = np.array([[y, x, 2.0, i] for i, (y, x) in enumerate(pts_ref)], dtype=np.float64)
        curr_blobs = np.array([[y + dy_obj, x + dx_obj, 2.0] for (y, x) in pts_ref], dtype=np.float64)
        tracked, _ = track_blobs(prev_blobs, curr_blobs, max_distance=2.0, next_id=3, drift_yx=(sy, sx))
        self.assertEqual([int(v) for v in tracked[:, 3]], [0, 1, 2])
        np.testing.assert_allclose(tracked[:, :2], curr_blobs[:, :2])

    def test_hungarian_strictly_one_to_one(self) -> None:
        prev = np.array([[10.0, 10.0, 3.0, 7.0]])
        curr = np.array([[10.5, 10.0, 3.0], [11.5, 10.0, 3.0]])
        tracked, next_id = track_blobs(prev, curr, max_distance=4.0, next_id=8)
        self.assertEqual([int(v) for v in tracked[:, 3]], [7, 8])
        self.assertEqual(next_id, 9)

    def test_bimodal_lspr_spectrum_analysis(self) -> None:
        wl = np.linspace(400.0, 1000.0, 481)
        bimodal = (
            120.0 * np.exp(-0.5 * ((wl - 520.0) / 25.0) ** 2)
            + 85.0 * np.exp(-0.5 * ((wl - 680.0) / 35.0) ** 2)
        )
        m, _, _, _, _ = analyze_spectrum(bimodal, wl, SpectraConfig(smoothing_window=21, smoothing_poly=3))
        self.assertGreaterEqual(m.Num_Peaks, 2)
        self.assertIsNotNone(m.Secondary_Peak_nm)
        assert m.Secondary_Peak_nm is not None
        self.assertAlmostEqual(m.Secondary_Peak_nm, 680.0, delta=6.0)
        self.assertFalse(m.Low_SNR)

    def test_multimodal_aggregate_classifier_and_no_nan_labels(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "Blob_ID": 0, "Frame_Number": 0, "Filename": "f0", "X": 50.0, "Y": 50.0, "Radius": 2.0,
                    "Spot_Area_px": 5, "Eccentricity": 0.2,
                    "Hue_mean_deg": 120.0, "Sat_mean": 0.8, "Val_mean": 0.5,
                    "FWHM_nm": 85.0, "Max_Wavelength_nm": 520.0, "Max_Intensity": 120.0,
                    "Num_Peaks": 1, "Low_SNR": False,
                },
                {
                    "Blob_ID": 1, "Frame_Number": 0, "Filename": "f0", "X": 100.0, "Y": 100.0, "Radius": 4.0,
                    "Spot_Area_px": 25, "Eccentricity": 0.78,
                    "Hue_mean_deg": 55.0, "Sat_mean": 0.5, "Val_mean": 0.9,
                    "FWHM_nm": 210.0, "Max_Wavelength_nm": 640.0, "Max_Intensity": 600.0,
                    "Num_Peaks": 2, "Low_SNR": False,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            agg_df = classify_aggregates(df, tmp, AggregateConfig(material_preset="agnp"))
            self.assertEqual(agg_df.iloc[0]["Aggregate_Class"], "Single")
            self.assertEqual(agg_df.iloc[0]["Aggregate_Evidence"], "Single")
            self.assertFalse(bool(agg_df.iloc[0]["Aggregate"]))

            self.assertEqual(agg_df.iloc[1]["Aggregate_Class"], "Aggregate")
            self.assertTrue(bool(agg_df.iloc[1]["Aggregate"]))
            self.assertIn("BroadFWHM", agg_df.iloc[1]["Aggregate_Evidence"])
            self.assertIn("RedShifted", agg_df.iloc[1]["Aggregate_Evidence"])
            self.assertIn("MultiPeak", agg_df.iloc[1]["Aggregate_Evidence"])

            reloaded = pd.read_csv(Path(tmp) / "postprocess" / "blobs_aggregate.csv")
            self.assertFalse(reloaded["Aggregate_Evidence"].isna().any())

        self.assertEqual(coarse_color_name(np.array([250, 20, 20]), gray=False), "Red")


class TestSyntheticEndToEndCI(unittest.TestCase):
    """Self-contained end-to-end tests that run in GitHub Actions without raw test_data/."""

    def test_qc_rejects_noisy_frame_and_accepts_clean_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds = _create_tiny_synthetic_dataset(tmp_path / "ds")
            clean_frames = discover_inputs(ds)

            rng = np.random.default_rng(7)
            noisy_img = rng.integers(85, 225, size=(128, 128, 3), dtype=np.uint8)
            noisy_tif = tmp_path / "bad_noisy_frame.tif"
            cv2.imwrite(str(noisy_tif), noisy_img)

            all_frames = list(clean_frames) + [FrameInput(stem="bad_noisy_frame", tiff_path=noisy_tif)]
            accepted, qc_df = run_frame_qc(all_frames, tmp_path / "qc_out", PipelineConfig().qc)
            self.assertEqual(len(accepted), 2)
            bad_row = qc_df[qc_df["Filename"] == "bad_noisy_frame"].iloc[0]
            self.assertFalse(bool(bad_row["Passed_QC"]))
            self.assertNotEqual(bad_row["Reject_Reason"], "Pass")

    def test_full_pipeline_report_cli_and_batch_on_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ds = _create_tiny_synthetic_dataset(tmp_path / "synthetic_exp")

            # Verify ENVI brace-aware header parsing (no leaked nested keys)
            frames = discover_inputs(ds)
            assert frames[0].envi_header_path is not None and frames[0].envi_data_path is not None
            hdr = parse_envi_header(frames[0].envi_header_path)
            self.assertEqual((hdr.samples, hdr.lines, hdr.bands), (128, 128, 61))
            assert hdr.raw_metadata is not None
            self.assertNotIn("date", hdr.raw_metadata)
            with EnviCube(frames[0].envi_header_path, frames[0].envi_data_path) as cube:
                self.assertEqual(cube.get_pixel_spectrum(36, 42).shape, (61,))

            # Run full Stages 0–5 pipeline
            out_dir = tmp_path / "run_out"
            cfg = load_config(DEFAULT_CONFIG_PATH)
            cfg.spectra.smoothing_window = 11
            cfg.spectra.smoothing_poly = 3
            cfg.spectra.peak_min_distance_bands = 5
            cfg.spectra.workers = 1
            summary = run_pipeline(ds, out_dir, cfg)

            self.assertGreaterEqual(len(summary), 4)
            # Confirm both Single and Aggregate classes were separated physically
            classes = set(summary["Aggregate_Class"].unique())
            self.assertIn("Single", classes)
            self.assertIn("Aggregate", classes)

            reloaded = pd.read_csv(out_dir / "blobs_summary.csv")
            self.assertFalse(reloaded["Aggregate_Evidence"].isna().any())

            # Verify drift & tracked shifts
            drift_df = pd.read_csv(out_dir / "tracking" / "frame_drift.csv")
            self.assertAlmostEqual(float(drift_df.iloc[1]["Shift_Y_px"]), 1.0, delta=0.5)
            self.assertAlmostEqual(float(drift_df.iloc[1]["Shift_X_px"]), -2.0, delta=0.5)
            shifts_df = pd.read_csv(out_dir / "tracking" / "tracked_shifts.csv")
            self.assertGreaterEqual(len(shifts_df), 2)

            # Verify per-blob CSVs, Blob_999-style plots, crops & FOV PNGs
            r0 = reloaded.iloc[0]
            for col in ("Snapshot_Normal", "Snapshot_Marked", "Spectrum_CSV", "Spectrum_Plot"):
                self.assertTrue((out_dir / str(r0[col])).is_file(), f"Missing {col}")
            self.assertTrue((out_dir / "reports" / "fov" / "sample_t0.png").is_file())

            # Verify standalone `darkfield-hsi report` CLI regeneration & HTML workstation payload
            cfg_path = tmp_path / "ci_cfg.yaml"
            save_config(cfg, cfg_path)
            self.assertEqual(
                cli_main(["report", "-i", str(ds), "-o", str(out_dir), "-c", str(cfg_path)]),
                0,
            )
            html_text = (out_dir / "reports" / "report.html").read_text(encoding="utf-8")
            for required_id in (
                "inspector-modal",
                "modal-prev",
                "modal-next",
                "modal-fov-canvas",
                "modal-note",
                "auto-advance",
                "fov-canvas",
                "scatter-canvas",
                "f-review",
                "f-evidence",
                "f-multipeak",
                "btn-import",
                "btn-export-filtered",
                "img-lightbox",
            ):
                self.assertIn(required_id, html_text)

            m_blob = re.search(r'<script id="blob-data" type="application/json">(.*?)</script>', html_text, re.S)
            m_fov = re.search(r'<script id="fov-data" type="application/json">(.*?)</script>', html_text, re.S)
            assert m_blob is not None and m_fov is not None
            self.assertEqual(len(json.loads(m_blob.group(1))), len(summary))
            self.assertIn("sample_t0", json.loads(m_fov.group(1)))

            # Verify TIFF-only fallback & batch mode
            batch_in = tmp_path / "batch_in"
            src_tif = ds / "tiff" / "sample_t0.tif"
            for name in ("exp_A", "exp_B"):
                sdir = batch_in / name
                sdir.mkdir(parents=True)
                shutil.copy2(src_tif, sdir / src_tif.name)
            batch_out = tmp_path / "batch_out"
            batch_df = run_batch(batch_in, batch_out, cfg)
            self.assertEqual(set(batch_df["Sample"].unique()), {"exp_A", "exp_B"})
            self.assertNotIn("Max_Band", batch_df.columns)
            self.assertTrue((batch_out / "batch_summary.csv").is_file())
            self.assertTrue((batch_out / "batch_overview.csv").is_file())

            # Verify `darkfield-hsi qc` CLI subcommand
            self.assertEqual(
                cli_main(["qc", "-i", str(ds), "-o", str(tmp_path / "qc_cli"), "-c", str(cfg_path)]),
                0,
            )


@unittest.skipUnless(TEST_DATA_DIR.is_dir(), "Local test_data/ not present")
class TestRealCytovivaData(unittest.TestCase):
    """Integration tests against local CytoViva datasets (auto-skipped in CI)."""

    def test_real_destriping_hdr_and_qc(self) -> None:
        frames = discover_inputs(TEST_DATA_DIR)
        self.assertEqual(len(frames), 2)
        assert frames[0].envi_header_path is not None and frames[0].envi_data_path is not None
        hdr = parse_envi_header(frames[0].envi_header_path)
        self.assertEqual((hdr.samples, hdr.lines, hdr.bands), (696, 696, 481))

        img = cv2.imread(str(frames[0].tiff_path), cv2.IMREAD_UNCHANGED)
        cleaned, _ = prepare_detection_image(img, PipelineConfig().detection)
        self.assertEqual(cleaned.shape, (696, 696))

        with tempfile.TemporaryDirectory() as tmp:
            accepted, _ = run_frame_qc(frames, tmp, PipelineConfig().qc)
            self.assertEqual(len(accepted), 2)

    def test_full_v02_pipeline_on_real_test_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "run_real"
            cfg = load_config(DEFAULT_CONFIG_PATH)
            # Disable per-blob plots here (already tested in synthetic CI) to keep real test crisp
            cfg.spectra.save_plots = False
            summary = run_pipeline(TEST_DATA_DIR, out_dir, cfg)

            self.assertGreater(len(summary), 0)
            detected = pd.read_csv(out_dir / "blobs_detected.csv")
            self.assertEqual(len(summary), len(detected))
            for _, grp in summary.groupby("Frame_Number"):
                self.assertEqual(grp["Blob_ID"].nunique(), len(grp))

            reloaded = pd.read_csv(out_dir / "blobs_summary.csv")
            self.assertFalse(reloaded["Aggregate_Evidence"].isna().any())
            self.assertTrue((out_dir / "reports" / "report.html").is_file())
            self.assertTrue((out_dir / "tracking" / "tracked_shifts.csv").is_file())


if __name__ == "__main__":
    unittest.main()
