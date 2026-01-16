"""
Basic Landsat 8/9 Analysis Example

This example demonstrates the simplest way to analyze Landsat 8/9 imagery
using default settings.
"""

import logging
from tocantins_framework import calculate_tocantins_framework

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

TIF_PATH = "data/LC08_L2SP_example.tif"
OUTPUT_DIR = "results/landsat8_basic"

logger.info("Starting Landsat 8/9 analysis")

calculator = calculate_tocantins_framework(
    tif_path=TIF_PATH,
    output_dir=OUTPUT_DIR
)

feature_set = calculator.get_feature_set()
impact_scores = calculator.get_impact_scores()
severity_scores = calculator.get_severity_scores()

logger.info(f"Analysis complete: {len(feature_set)} anomalies detected")

if len(feature_set) > 0:
    top_anomalies = feature_set.nlargest(5, 'IS')[['Anomaly_ID', 'Type', 'IS', 'SS']]
    logger.info(f"Top 5 anomalies by Impact Score:\n{top_anomalies}")

logger.info(f"Results saved to: {OUTPUT_DIR}/")
