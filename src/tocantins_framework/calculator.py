"""
A 'calculator' that runs the complete Tocantins Framework analysis pipeline.

Integrates preprocessing, anomaly detection, morphology, metrics, and I/O.
"""

import logging
from typing import Optional, Dict

import pandas as pd
import numpy as np

from .preprocessing import LandsatPreprocessor
from .anomaly_detection import AnomalyDetector
from .morphology import MorphologyProcessor
from .metrics import MetricsCalculator
from .io import ResultsWriter

logger = logging.getLogger(__name__)


class TocantinsFrameworkCalculator:
    """
    Main calculator for the Tocantins Framework analysis.
    
    Orchestrates the pipeline from raw Landsat imagery to Impact Score (IS)
    and Severity Score (SS) metrics for urban heat anomaly characterization.
    """
    
    def __init__(
        self,
        rf_params: Optional[Dict] = None,
        k_threshold: float = 1.5,
        spatial_params: Optional[Dict] = None,
        impact_params: Optional[Dict] = None,
        severity_params: Optional[Dict] = None
    ):
        self.k_threshold = k_threshold
        
        self.preprocessor = LandsatPreprocessor()
        self.detector = AnomalyDetector(k_threshold, rf_params)
        self.morph_processor = MorphologyProcessor(spatial_params)
        self.metrics = MetricsCalculator(impact_params or severity_params)
        
        self.full_data = None
        self.raster_meta = {}
        self.impact_scores = None
        self.severity_scores = None
        self.feature_set = None
        
        self._m1_hot_2d = None
        self._m1_cold_2d = None
        self._residual_2d = None
        self._unified_hot_cores = None
        self._unified_cold_cores = None
        self._coherent_hot_eaz = None
        self._coherent_cold_eaz = None
        self._zone_classification = None
    
    def run_complete_analysis(
        self,
        tif_path: str,
        output_dir: str = "tocantins_framework_results",
        save_results: bool = True
    ) -> bool:
        """Execute complete Tocantins Framework analysis pipeline."""
        logger.info("Starting Tocantins Framework analysis")
        
        try:
            self._run_pipeline(tif_path)
            
            if save_results:
                self._save_all_results(output_dir)
            
            logger.info("Analysis completed successfully")
            self._log_summary()
            
            return True
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return False
    
    def _run_pipeline(self, tif_path: str) -> None:
        self.full_data, self.raster_meta = self.preprocessor.load_imagery(tif_path)
        
        lst_2d = self.preprocessor.get_lst_2d()
        valid_mask_2d = self.preprocessor.get_valid_mask_2d()
        
        self._m1_hot_2d, self._m1_cold_2d, self.full_data = \
            self.detector.detect_statistical_anomalies(self.full_data, lst_2d, valid_mask_2d)
        
        self.detector.train_model(self.full_data)
        
        self._residual_2d, self.full_data = \
            self.detector.calculate_residuals(self.full_data, lst_2d)
        
        core_hot, core_cold = self.detector.refine_anomaly_cores(
            self._m1_hot_2d, self._m1_cold_2d, self._residual_2d, valid_mask_2d
        )
        
        self._unified_hot_cores, self._unified_cold_cores, _, _ = \
            self.morph_processor.create_unified_cores(core_hot, core_cold)
        
        training_stats = self.detector.get_training_stats()
        
        self._coherent_hot_eaz, self._coherent_cold_eaz = \
            self.morph_processor.grow_eaz(
                self._unified_hot_cores, self._unified_cold_cores,
                self._residual_2d, valid_mask_2d,
                training_stats['residual_std'], self.k_threshold
            )
        
        self._zone_classification = self.morph_processor.create_classification_map(
            lst_2d.shape,
            self._coherent_cold_eaz, self._coherent_hot_eaz,
            self._unified_cold_cores, self._unified_hot_cores
        )
        
        spatial_params = self.morph_processor.params
        pixel_size = self.raster_meta.get('pixel_size', 30.0)
        connectivity = spatial_params['connectivity']
        
        self.impact_scores = self.metrics.calculate_impact_scores(
            self._unified_hot_cores, self._unified_cold_cores,
            self._coherent_hot_eaz, self._coherent_cold_eaz,
            self._residual_2d, training_stats['residual_std'],
            pixel_size, connectivity
        )
        
        self.severity_scores = self.metrics.calculate_severity_scores(
            self._unified_hot_cores, self._unified_cold_cores,
            self._residual_2d, training_stats['residual_std'],
            pixel_size, connectivity
        )
        
        self._merge_feature_set()
    
    def _merge_feature_set(self) -> None:
        """Merge Impact and Severity scores into unified feature set."""
        if self.impact_scores is None or self.impact_scores.empty:
            self.feature_set = pd.DataFrame()
            return
        
        if self.severity_scores is None or self.severity_scores.empty:
            self.feature_set = pd.DataFrame()
            return
        
        merged = pd.merge(
            self.impact_scores,
            self.severity_scores,
            on=['Anomaly_ID', 'Type'],
            suffixes=('', '_core')
        )
        
        merged = merged.rename(columns={
            'Centroid_Row_core': 'Core_Centroid_Row',
            'Centroid_Col_core': 'Core_Centroid_Col',
            'Area_m2': 'EAZ_Area_m2',
            'Area_pixels': 'EAZ_Area_pixels'
        })
        
        self.feature_set = merged
    
    def _save_all_results(self, output_dir: str) -> None:
        writer = ResultsWriter(self.raster_meta)
        writer.save_all(output_dir, self.impact_scores, self._zone_classification, self._residual_2d)
        
        from pathlib import Path
        output_path = Path(output_dir)
        
        if self.severity_scores is not None and not self.severity_scores.empty:
            severity_path = output_path / "severity_scores.csv"
            self.severity_scores.to_csv(severity_path, index=False, float_format='%.6f')
            logger.info(f"Saved Severity Scores: {severity_path}")
        
        if self.feature_set is not None and not self.feature_set.empty:
            features_path = output_path / "ml_features.csv"
            self.feature_set.to_csv(features_path, index=False, float_format='%.6f')
            logger.info(f"Saved ML feature set: {features_path}")
    
    def _log_summary(self) -> None:
        if self.feature_set is not None and not self.feature_set.empty:
            logger.info(f"Analysis complete: {len(self.feature_set)} anomalies detected")
    
    def get_feature_set(self) -> pd.DataFrame:
        """Get complete feature set with both IS and SS metrics."""
        return self.feature_set
    
    def get_impact_scores(self) -> pd.DataFrame:
        return self.impact_scores
    
    def get_severity_scores(self) -> pd.DataFrame:
        return self.severity_scores
    
    def get_classification_map(self):
        return self._zone_classification
    
    def get_residual_map(self):
        return self._residual_2d


def calculate_tocantins_framework(
    tif_path: str,
    output_dir: str = "output",
    spatial_params: Optional[Dict] = None,
    k_threshold: float = 1.5,
    rf_params: Optional[Dict] = None,
    impact_params: Optional[Dict] = None,
    severity_params: Optional[Dict] = None
) -> TocantinsFrameworkCalculator:
    """
    Calculate Impact Score and Severity Score for thermal anomalies in Landsat imagery.
    
    Args:
        tif_path: Path to Landsat GeoTIFF file.
        output_dir: Output directory path.
        spatial_params: Spatial processing parameters.
        k_threshold: Threshold multiplier for residual detection.
        rf_params: Random Forest model parameters.
        impact_params: Impact score calculation parameters.
        severity_params: Severity score calculation parameters.
        
    Returns:
        TocantinsFrameworkCalculator instance with computed results.
    """
    calculator = TocantinsFrameworkCalculator(
        k_threshold=k_threshold,
        spatial_params=spatial_params,
        rf_params=rf_params,
        impact_params=impact_params,
        severity_params=severity_params
    )
    calculator.run_complete_analysis(tif_path, output_dir, save_results=True)
    return calculator
