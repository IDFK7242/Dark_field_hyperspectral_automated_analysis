"""Automated dark-field hyperspectral imaging (HSI) processing package."""

from darkfield_hsi.config import PipelineConfig, load_config, save_config
from darkfield_hsi.pipeline import run_batch, run_pipeline

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "PipelineConfig",
    "load_config",
    "save_config",
    "run_pipeline",
    "run_batch",
]
