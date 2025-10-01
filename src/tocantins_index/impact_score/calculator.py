"""
A 'calculator' that runs the complete analysis pipeline.

It integrates preprocessing, anomaly detection, morphology, metrics, and I/O.
"""

import logging
from typing import Optional, Dict

import pandas as pd

from .preprocessing import LandsatPreprocessor
from .anomaly_detection import AnomalyDetector
from .morphology import MorphologyProcessor
from .metrics import ImpactScoreMetrics
from .io import ResultsWriter

logger = logging.getLogger(__name__)


class ImpactScoreCalculator:
    """
    Main calculator for Impact Score analysis of thermal anomalies.
    
    Orchestrates the complete pipeline from raw Landsat imagery to
    quantified anomaly impact scores.
    
    Attributes:
        k_threshold: Threshold multiplier for residual-based anomaly detection.
        rf_params: Random Forest model configuration.
        spatial_params: Spatial processing parameters for morphology operations.
        impact_params: Impact score calculation parameters.
    """
    
    def __init__(
        self,
        rf_params: Optional[Dict] = None,
        k_threshold: float = 1.5,
        spatial_params: Optional[Dict] = None,
        impact_params: Optional[Dict] = None
    ):
        """
        Initialize Impact Score Calculator.
        
        Args:
            rf_params: Random Forest model parameters.
            k_threshold: Threshold multiplier for residual detection.
            spatial_params: Spatial processing parameters.
            impact_params: Impact score calculation parameters.
        """
        self.k_threshold = k_threshold
        
        self.preprocessor = LandsatPreprocessor()
        self.detector = AnomalyDetector(k_threshold, rf_params)
        self.morph_processor = MorphologyProcessor(spatial_params)
        self.metrics_calculator = ImpactScoreMetrics(impact_params)
        
        self.full_data = None
        self.raster_meta = {}
        self.impact_scores = None
        
        self._m1_hot_2d = None
        self._m1_cold_2d = None
        self._residual_2d = None
        self._unified_hot_cores = None
        self._unified_cold_cores = None
        self._coherent_hot_influence = None
        self._coherent_cold_influence = None
        self._zone_classification = None
    
    def run_complete_analysis(
        self,
        tif_path: str,
        output_dir: str = "impact_score_results",
        save_results: bool = True
    ) -> bool:
        """
        Execute complete analysis pipeline.
        
        Args:
            tif_path: Path to input Landsat GeoTIFF.
            output_dir: Output directory path.
            save_results: Whether to save results to disk.
            
        Returns:
            True if analysis completed successfully, False otherwise.
        """
        logger.info("Starting Impact Score analysis")
        
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
        """
        Execute analysis pipeline steps.
        
        Args:
            tif_path: Path to input Landsat GeoTIFF.
        """
        self.full_data, self.raster_meta = self.preprocessor.load_imagery(tif_path)
        
        lst_2d = self.preprocessor.get_lst_2d()
        valid_mask_2d = self.preprocessor.get_valid_mask_2d()
        
        self._m1_hot_2d, self._m1_cold_2d, self.full_data = \
            self.detector.detect_statistical_anomalies(
                self.full_data, lst_2d, valid_mask_2d
            )
        
        self.detector.train_model(self.full_data)
        
        self._residual_2d, self.full_data = \
            self.detector.calculate_residuals(self.full_data, lst_2d)
        
        core_hot, core_cold = self.detector.refine_anomaly_cores(
            self._m1_hot_2d,
            self._m1_cold_2d,
            self._residual_2d,
            valid_mask_2d
        )
        
        self._unified_hot_cores, self._unified_cold_cores, _, _ = \
            self.morph_processor.create_unified_cores(core_hot, core_cold)
        
        training_stats = self.detector.get_training_stats()
        
        self._coherent_hot_influence, self._coherent_cold_influence = \
            self.morph_processor.grow_influence_zones(
                self._unified_hot_cores,
                self._unified_cold_cores,
                self._residual_2d,
                valid_mask_2d,
                training_stats['residual_std'],
                self.k_threshold
            )
        
        self._zone_classification = self.morph_processor.create_classification_map(
            lst_2d.shape,
            self._coherent_cold_influence,
            self._coherent_hot_influence,
            self._unified_cold_cores,
            self._unified_hot_cores
        )
        
        spatial_params = self.morph_processor.params
        
        self.impact_scores = self.metrics_calculator.calculate_scores(
            self._unified_hot_cores,
            self._unified_cold_cores,
            self._coherent_hot_influence,
            self._coherent_cold_influence,
            self._residual_2d,
            training_stats['residual_std'],
            self.raster_meta.get('pixel_size', 30.0),
            spatial_params['connectivity']
        )
    
    def _save_all_results(self, output_dir: str) -> None:
        """
        Save all analysis results.
        
        Args:
            output_dir: Output directory path.
        """
        writer = ResultsWriter(self.raster_meta)
        writer.save_all(
            output_dir,
            self.impact_scores,
            self._zone_classification,
            self._residual_2d
        )
    
    def _log_summary(self) -> None:
        """Log analysis summary."""
        if self.impact_scores is not None and not self.impact_scores.empty:
            logger.info("\nTop 5 Anomalies:")
            summary_cols = ['Anomaly_ID', 'Type', 'IS', 'Severity', 'Area_m2', 'Coherence']
            print(self.impact_scores[summary_cols].head())
    
    def get_impact_scores(self) -> pd.DataFrame:
        """
        Get calculated Impact Scores.
        
        Returns:
            DataFrame with Impact Scores and submetrics.
        """
        return self.impact_scores
    
    def get_classification_map(self):
        """Get zone classification map."""
        return self._zone_classification
    
    def get_residual_map(self):
        """Get LST residual map."""
        return self._residual_2d


def calculate_impact_scores(
    tif_path: str,
    output_dir: str = "output",
    spatial_params: Optional[Dict] = None,
    k_threshold: float = 1.5,
    rf_params: Optional[Dict] = None,
    impact_params: Optional[Dict] = None
) -> ImpactScoreCalculator:
    """
    Calculate Impact Scores for thermal anomalies in Landsat imagery.
    
    A convenience function that creates a calculator and runs the complete analysis.
    
    Args:
        tif_path: Path to Landsat GeoTIFF file.
        output_dir: Output directory path.
        spatial_params: Spatial processing parameters.
        k_threshold: Threshold multiplier for residual detection.
        rf_params: Random Forest model parameters.
        impact_params: Impact score calculation parameters.
        
    Returns:
        ImpactScoreCalculator instance with computed results.
    """
    calculator = ImpactScoreCalculator(
        k_threshold=k_threshold,
        spatial_params=spatial_params,
        rf_params=rf_params,
        impact_params=impact_params
    )
    calculator.run_complete_analysis(tif_path, output_dir, save_results=True)
    return calculator
