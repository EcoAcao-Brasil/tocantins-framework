"""
Setup package to calculate the Impact Score.

A toolkit for detecting and quantifying the thermo-spatial impact
Intra-urban Heat Anomalies play over its surroundings (zones of influence)
in Landsat imagery using machine learning and spatial analysis techniques.

This metric is a subscore of the compounded Tocantins Index.
"""

from .calculator import ImpactScoreCalculator, calculate_impact_scores

__version__ = "1.0.0"
__author__ = "Isaque Carvalho Borges"

__all__ = [
    "ImpactScoreCalculator",
    "calculate_impact_scores",
]
