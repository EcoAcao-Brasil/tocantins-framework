"""
Advanced Configuration Example

This example demonstrates full control over all framework parameters
for specialized analysis scenarios.
"""

import logging
from tocantins_framework import TocantinsFrameworkCalculator

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

SPATIAL_PARAMS = {
    'min_anomaly_size': 5,
    'agglutination_distance': 3,
    'morphology_kernel_size': 3,
    'connectivity': 2
}

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 30,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

METRICS_PARAMS = {
    'min_eaz_pixels': 3,
    'min_core_pixels': 2,
    'std_floor_degC': 0.1
}

K_THRESHOLD = 2.0

logger.info("Initializing calculator with custom parameters")
logger.debug(f"k_threshold: {K_THRESHOLD}")
logger.debug(f"RF n_estimators: {RF_PARAMS['n_estimators']}")
logger.debug(f"Min anomaly size: {SPATIAL_PARAMS['min_anomaly_size']} pixels")

calculator = TocantinsFrameworkCalculator(
    band_mapping=None,
    rf_params=RF_PARAMS,
    k_threshold=K_THRESHOLD,
    spatial_params=SPATIAL_PARAMS,
    impact_params=METRICS_PARAMS,
    severity_params=METRICS_PARAMS
)

TIF_PATH = "data/LC08_L2SP_example.tif"
OUTPUT_DIR = "results/advanced_analysis"

success = calculator.run_complete_analysis(
    tif_path=TIF_PATH,
    output_dir=OUTPUT_DIR,
    save_results=True
)

if success:
    feature_set = calculator.get_feature_set()
    classification_map = calculator.get_classification_map()
    residual_map = calculator.get_residual_map()
    
    logger.info(f"Total anomalies detected: {len(feature_set)}")
    logger.info(f"Classification map shape: {classification_map.shape}")
    logger.info(f"Residual map range: [{residual_map.min():.2f}, {residual_map.max():.2f}]°C")
    
    logger.info("Impact Score statistics:")
    logger.info(f"  Mean: {feature_set['IS'].mean():.3f}")
    logger.info(f"  Median: {feature_set['IS'].median():.3f}")
    logger.info(f"  Std: {feature_set['IS'].std():.3f}")
    
    logger.info("Severity Score statistics:")
    logger.info(f"  Mean: {feature_set['SS'].mean():.3f}")
    logger.info(f"  Median: {feature_set['SS'].median():.3f}")
    logger.info(f"  Std: {feature_set['SS'].std():.3f}")
    
    logger.info(f"Outputs saved to: {OUTPUT_DIR}/")
else:
    logger.error("Analysis failed - check logs for details")
