"""
The Tocantins Index Framework Package

A feature extraction toolkit for detecting and quantifying intra-urban heat anomalies 
in Landsat imagery using machine learning and spatial analysis.

The framework produces two complementary metrics:
- Impact Score (IS): Quantifies thermal influence over surroundings
- Severity Score (SS): Quantifies thermal anomalousness of the core

These metrics provide rich features for AI-driven urban heat intervention planning.

Components:
- Preprocessing: Landsat band extraction and spectral index calculation
- Anomaly Detection: Statistical and ML-based thermal anomaly detection
- Morphology: Spatial operations for cores and influence zones
- Metrics: Impact Score and Severity Score calculation
- Calculator: Complete feature extraction pipeline
"""

from .calculator import TocantinsFrameworkCalculator, calculate_tocantins_framework
from .preprocessing import LandsatPreprocessor
from .anomaly_detection import AnomalyDetector
from .morphology import MorphologyProcessor
from .metrics import MetricsCalculator
from .io import ResultsWriter

__version__ = "1.0.0"
__author__ = "Isaque Carvalho Borges"

__all__ = [
    "TocantinsIndexCalculator",
    "calculate_tocantins_index",
    "LandsatPreprocessor",
    "AnomalyDetector",
    "MorphologyProcessor",
    "MetricsCalculator",
    "ResultsWriter",
]
