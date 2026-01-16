"""
Tocantins Framework: A Python Library for Assessment of Intra-Urban Thermal Anomaly

A comprehensive Python framework for detecting and quantifying intra-urban 
thermal anomalies in Landsat imagery using machine learning and spatial analysis.

Basic Usage:
>>> from tocantins_framework import calculate_tocantins_framework
>>> 
>>> # Landsat 8/9 (default)
>>> calculator = calculate_tocantins_framework(
...     tif_path="path/to/landsat_scene.tif",
...     output_dir="results"
... )
>>> 
>>> # Landsat 5
>>> l5_mapping = {
...     'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3',
...     'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7',
...     'thermal': 'ST_B6', 'qa_pixel': 'QA_PIXEL'
... }
>>> calculator = calculate_tocantins_framework(
...     tif_path="path/to/landsat5_scene.tif",
...     band_mapping=l5_mapping,
...     output_dir="results"
... )
"""

from .calculator import TocantinsFrameworkCalculator, calculate_tocantins_framework
from .preprocessing import LandsatPreprocessor
from .anomaly_detection import AnomalyDetector
from .morphology import MorphologyProcessor
from .metrics import MetricsCalculator
from .io import ResultsWriter

__version__ = "1.1.0"
__author__ = "Isaque Carvalho Borges"
__email__ = "isaque@ecoacaobrasil.org"
__organization__ = "EcoAção Brasil"
__license__ = "MIT"

__all__ = [
    "TocantinsFrameworkCalculator",
    "calculate_tocantins_framework",
    "LandsatPreprocessor",
    "AnomalyDetector",
    "MorphologyProcessor",
    "MetricsCalculator",
    "ResultsWriter",
    "__version__",
    "__author__",
]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())


def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for the Tocantins Framework.
    
    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO).
    log_file : str, optional
        Path to log file. If None, logs only to console.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger
