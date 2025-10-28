"""
The Tocantins Index Framework Package

A toolkit for comprehensively detecting and quantifying
Intra-urban Heat Anomalies in Landsat imagery using 
Machine Learning and spatial analysis techniques.

The Tocantins Index is the result of integrating the Impact Score and the Severity Score.
"""


from .calculator import TocantinsIndexCalculator, calculate_tocantins_index

__version__ = "1.0.0"
__author__ = "Isaque Carvalho Borges"

__all__ = [
    "TocantinsIndexCalculator",
    "calculate_tocantins_index",
]
