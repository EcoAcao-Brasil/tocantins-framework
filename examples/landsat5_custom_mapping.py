"""
Landsat 5 Analysis with Custom Band Mapping

This example shows how to analyze Landsat 5 imagery by providing
a custom band mapping configuration.
"""

import logging
from tocantins_framework import calculate_tocantins_framework

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

L5_BAND_MAPPING = {
    'blue': 'SR_B1',
    'green': 'SR_B2',
    'red': 'SR_B3',
    'nir': 'SR_B4',
    'swir1': 'SR_B5',
    'swir2': 'SR_B7',
    'thermal': 'ST_B6',
    'qa_pixel': 'QA_PIXEL'
}

TIF_PATH = "data/LT05_L2SP_example.tif"
OUTPUT_DIR = "results/landsat5_analysis"

logger.info("Starting Landsat 5 analysis with custom band mapping")

calculator = calculate_tocantins_framework(
    tif_path=TIF_PATH,
    band_mapping=L5_BAND_MAPPING,
    output_dir=OUTPUT_DIR
)

feature_set = calculator.get_feature_set()

logger.info(f"Detected {len(feature_set)} thermal anomalies")

hot_anomalies = feature_set[feature_set['Type'] == 'hot']
cold_anomalies = feature_set[feature_set['Type'] == 'cold']

logger.info(f"Hot anomalies: {len(hot_anomalies)}")
if len(hot_anomalies) > 0:
    logger.info(f"  Mean IS: {hot_anomalies['IS'].mean():.3f}, Mean SS: {hot_anomalies['SS'].mean():.3f}")

logger.info(f"Cold anomalies: {len(cold_anomalies)}")
if len(cold_anomalies) > 0:
    logger.info(f"  Mean IS: {cold_anomalies['IS'].mean():.3f}, Mean SS: {cold_anomalies['SS'].mean():.3f}")

logger.info(f"Results saved to: {OUTPUT_DIR}/")
