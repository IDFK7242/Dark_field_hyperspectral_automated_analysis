# Data Formats & Output Schemas

## Input Directory Layout

`darkfield-hsi` supports two directory layouts. Raw microscopy files are excluded from Git via `.gitignore` and should be kept locally (for example under `test_data/` or an external drive).

### Layout A — Separate `tiff/` and optional `envi/` subfolders (recommended)

```text
my_experiment/
├── tiff/
│   ├── sample_t0.tif
│   └── sample_t5min.tif
└── envi/                      # Optional (Stages 2–3 skipped cleanly if absent)
    ├── sample_t0              # Extensionless ENVI binary (or .raw/.img/.bil/.bsq/.bip/.dat)
    ├── sample_t0.hdr
    ├── sample_t5min
    └── sample_t5min.hdr
```

### Layout B — Co-located files in one directory

```text
my_experiment/
├── sample_t0.tif
├── sample_t0
└── sample_t0.hdr
```

### Batch Mode Layout (`darkfield-hsi batch`)

```text
experiments_root/
├── sample_A/
│   ├── tiff/
│   └── envi/
└── sample_B/
    ├── tiff/
    └── envi/
```

## ENVI Header (`.hdr`) Support

- Supported interleaves: `bsq`, `bil`, `bip`
- Supported ENVI data types: `1` (`uint8`), `2` (`int16`), `3` (`int32`), `4` (`float32`), `5` (`float64`), `12` (`uint16`), `13` (`uint32`), `14` (`int64`), `15` (`uint64`)
- Byte order: `0` (little-endian) and `1` (big-endian), plus non-zero `header offset`
- Multi-line `{ ... }` blocks (including `description = { ... }` containing `=` signs and calibrated `wavelength = { ... }` arrays in nanometers)

## Key Output CSV Schemas

All CSVs use UTF-8 encoding with comma (`,`) separators. Non-aggregate blobs use the explicit string `"Single"` (never `"None"`) so `pandas.read_csv()` and Excel do not coerce labels to `NaN`.

### `blobs_summary.csv` (Master Merged Table)

| Column Group | Columns |
| :--- | :--- |
| Identification & Location | `Blob_ID`, `Frame_Number`, `Filename`, `X`, `Y`, `Radius` |
| Spot Morphology (Stage 1) | `Spot_Peak_Intensity`, `Spot_Area_px`, `Spot_Radius_FWHM_px`, `Eccentricity`, `Is_Saturated` |
| Foreground Color & Crops (Stage 4a) | `Color`, `Hue_mean_deg`, `Hue_std_deg`, `Sat_mean`, `Val_mean`, `Fg_Pixel_Count`, `n_used_for_hue`, `Snapshot_Normal`, `Snapshot_Marked` |
| Spectral Metrics (Stage 3, when ENVI present) | `Max_Band`, `Max_Wavelength_nm`, `Max_Intensity`, `Half_Max_Intensity`, `Left_Half_Band`, `Right_Half_Band`, `FWHM_Bands`, `Left_Half_Wavelength_nm`, `Right_Half_Wavelength_nm`, `FWHM_nm`, `Spectral_Noise`, `Spectral_SNR`, `Low_SNR`, `Num_Peaks`, `Secondary_Peak_Band`, `Secondary_Peak_nm`, `Secondary_Peak_Intensity`, `Spectrum_CSV`, `Spectrum_Plot` |
| Multi-Modal Aggregates (Stage 4b) | `Aggregate_Score`, `Aggregate_Class`, `Aggregate`, `Aggregate_Evidence` |

### `spectra/per_blob_csv/<stem>/<stem>_b00000_spectrum.csv`

One CSV per detected nanoparticle, ready for direct plotting in Origin, Excel, or Python:

`Band_Index,Wavelength_nm,Raw_Intensity,Smoothed_Intensity,Normalized_Intensity`

### `qc/frame_qc.csv`

`Filename,Tiff_Path,Bg_Median,Bg_MAD,HF_Noise_MAD,Stripe_Index,SNR_P995,Laplacian_Focus,Passed_QC,Reject_Reason`

### `tracking/tracked_shifts.csv`

One row per `Blob_ID` tracked across `>= 2` frames:

`Blob_ID,Frames_Seen,First_Frame,Last_Frame,First_Filename,Last_Filename,X_start,Y_start,X_end,Y_end,Displacement_px,Color_start,Color_end,Aggregate_Class_start,Aggregate_Class_end,Aggregate_Score_start,Aggregate_Score_end,Peak_nm_start,Peak_nm_end,Delta_Peak_nm,FWHM_nm_start,FWHM_nm_end,Delta_FWHM_nm,Intensity_start,Intensity_end,Delta_Intensity`
