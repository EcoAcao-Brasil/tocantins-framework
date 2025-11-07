"""
Tocantins Framework: A Python Library for Assessment of Intra-Urban Thermal Anomaly

A comprehensive Python framework for detecting and quantifying intra-urban 
thermal anomalies in Landsat imagery using machine learning and spatial analysis.

The framework implements a scientifically rigorous methodology combining:
- Statistical anomaly detection
- Machine learning-based residual analysis (Random Forest)
- Spatial morphological operations
- Standardized impact and severity metrics

Key Concepts:
Extended Anomaly Zone (EAZ):
    Spatially coherent regions exhibiting thermal anomalies, extending beyond
    the immediate anomaly core to capture the full spatial impact.

Impact Score (IS):
    Quantifies the spatial extent and thermal significance of the EAZ.
    Formula: IS = sign(ΔT) × log(1 + severity × area × continuity)
    
Severity Score (SS):
    Quantifies the thermal intensity of the anomaly core itself.
    Formula: SS = sign(ΔT) × log(1 + thermal_intensity × area)

Basic Usage:
>>> from tocantins_framework import calculate_tocantins_framework
>>> 
>>> # Run complete analysis
>>> calculator = calculate_tocantins_framework(
...     tif_path="path/to/landsat_scene.tif",
...     output_dir="results"
... )
>>> 
>>> # Access results
>>> features = calculator.get_feature_set()
>>> print(features[['Anomaly_ID', 'Type', 'IS', 'SS']])

Advanced Usage:
>>> from tocantins_framework import TocantinsFrameworkCalculator
>>> 
>>> # Custom configuration
>>> calculator = TocantinsFrameworkCalculator(
...     k_threshold=1.5,
...     spatial_params={'min_anomaly_size': 5},
...     rf_params={'n_estimators': 300}
... )
>>> 
>>> # Run analysis with custom settings
>>> success = calculator.run_complete_analysis(
...     tif_path="path/to/landsat_scene.tif",
...     output_dir="results"
... )

Components:
TocantinsFrameworkCalculator : class
    Main orchestrator for complete thermal anomaly analysis pipeline.

calculate_tocantins_framework : function
    Convenience function for one-line complete analysis.

LandsatPreprocessor : class
    Handles Landsat imagery loading, band extraction, and spectral indices.

AnomalyDetector : class
    Implements statistical and ML-based thermal anomaly detection methods.

MorphologyProcessor : class
    Performs spatial morphological operations on anomaly masks.

MetricsCalculator : class
    Calculates Impact Score (IS) and Severity Score (SS) metrics.

ResultsWriter : class
    Handles export of analysis results to CSV and GeoTIFF formats.

Version Information
-------------------
:Version: 1.0.0
:Author: Isaque Carvalho Borges
:Organization: EcoAção Brasil
:License: MIT

See Also:
Repository: https://github.com/EcoAcao-Brasil/tocantins-framework
Organization Website: https://ecoacaobrasil.org
"""

from .calculator import TocantinsFrameworkCalculator, calculate_tocantins_framework
from .preprocessing import LandsatPreprocessor
from .anomaly_detection import AnomalyDetector
from .morphology import MorphologyProcessor
from .metrics import MetricsCalculator
from .io import ResultsWriter

# Version and metadata
__version__ = "1.0.0"
__author__ = "Isaque Carvalho Borges"
__email__ = "isaque@ecoacaobrasil.org"
__organization__ = "EcoAção Brasil"
__license__ = "MIT"

# Public API
__all__ = [
    # Main interface
    "TocantinsFrameworkCalculator",
    "calculate_tocantins_framework",
    
    # Processing components (for advanced users)
    "LandsatPreprocessor",
    "AnomalyDetector",
    "MorphologyProcessor",
    "MetricsCalculator",
    "ResultsWriter",
    
    # Metadata
    "__version__",
    "__author__",
]


# Module-level configuration
import logging

# Set up null handler to prevent "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())


def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for the Tocantins Framework.
    
    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.INFO).
        Use logging.DEBUG for detailed diagnostics.
    log_file : str, optional
        Path to log file. If None, logs only to console.
    
    Examples
    --------
    >>> import tocantins_framework as tf
    >>> tf.setup_logging(level=logging.DEBUG, log_file='analysis.log')
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger