"""
Impact Score calculation and metrics.

Implements the Impact Score metric and submetric calculations.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
from skimage import measure

logger = logging.getLogger(__name__)


class ImpactScoreMetrics:
    """
    Calculator for Impact Score and associated submetrics.
    """
    
    DEFAULT_PARAMS = {
        'min_influence_pixels': 1,
        'std_floor_degC': 0.05
    }
    
    def __init__(self, params: Dict = None):
        """
        Initialize metrics calculator.
        
        Args:
            params: Impact score calculation parameters.
        """
        self.params = params or self.DEFAULT_PARAMS.copy()
        self._gradient_magnitude_2d = None
    
    def compute_gradient_map(self, residual_2d: np.ndarray) -> None:
        """
        Calculates the spatial gradient magnitude of LST residuals.
        
        Args:
            residual_2d: 2D array of LST residuals.
        """
        grad_y, grad_x = np.gradient(np.nan_to_num(residual_2d))
        self._gradient_magnitude_2d = np.sqrt(grad_y**2 + grad_x**2)
    
    def calculate_scores(
        self,
        hot_cores: np.ndarray,
        cold_cores: np.ndarray,
        hot_influence: np.ndarray,
        cold_influence: np.ndarray,
        residual_2d: np.ndarray,
        residual_std: float,
        pixel_size_m: float,
        connectivity: int = 2
    ) -> pd.DataFrame:
        """
        Calculate Impact Scores for all detected anomalies.
        
        Args:
            hot_cores: Hot anomaly core mask.
            cold_cores: Cold anomaly core mask.
            hot_influence: Hot influence zone mask.
            cold_influence: Cold influence zone mask.
            residual_2d: LST residual array.
            residual_std: Standard deviation of residuals.
            pixel_size_m: Pixel size in meters.
            connectivity: Pixel connectivity for labeling.
            
        Returns:
            DataFrame with Impact Scores and submetrics for all anomalies.
        """
        logger.info("Calculating Impact Scores")
        
        self.compute_gradient_map(residual_2d)
        
        hot_scores = self._score_anomalies(
            hot_cores,
            hot_influence,
            'hot',
            residual_2d,
            residual_std,
            pixel_size_m,
            connectivity
        )
        
        cold_scores = self._score_anomalies(
            cold_cores,
            cold_influence,
            'cold',
            residual_2d,
            residual_std,
            pixel_size_m,
            connectivity
        )
        
        all_scores = pd.concat([hot_scores, cold_scores], ignore_index=True)
        
        if not all_scores.empty:
            all_scores = all_scores.sort_values(
                by='IS',
                key=abs,
                ascending=False
            ).reset_index(drop=True)
            logger.info(f"Calculated IS for {len(all_scores)} anomalies")
        else:
            logger.warning("No anomalies detected")
        
        return all_scores
    
    def _score_anomalies(
        self,
        cores_mask: np.ndarray,
        influence_mask: np.ndarray,
        anomaly_type: str,
        residual_2d: np.ndarray,
        residual_std: float,
        pixel_size_m: float,
        connectivity: int
    ) -> pd.DataFrame:
        """
        Calculate Impact Scores for all anomalies of a given type.
        
        Args:
            cores_mask: Binary mask of anomaly cores.
            influence_mask: Binary mask of influence zones.
            anomaly_type: Type of anomaly ('hot' or 'cold').
            residual_2d: LST residual array.
            residual_std: Standard deviation of residuals.
            pixel_size_m: Pixel size in meters.
            connectivity: Pixel connectivity for labeling.
            
        Returns:
            DataFrame with Impact Scores and submetrics.
        """
        if not np.any(cores_mask):
            return pd.DataFrame()
        
        labeled_cores = measure.label(cores_mask, connectivity=connectivity)
        full_anomalies_mask = cores_mask | influence_mask
        labeled_full = measure.label(full_anomalies_mask, connectivity=connectivity)
        
        core_regions = measure.regionprops(labeled_cores)
        
        results = []
        for core_region in core_regions:
            cy, cx = core_region.centroid
            full_label = labeled_full[int(cy), int(cx)]
            
            if full_label == 0:
                continue
            
            full_mask = (labeled_full == full_label)
            influence_only = full_mask & influence_mask
            
            score_dict = self._calculate_single_score(
                residual_2d,
                full_mask,
                influence_only,
                residual_std,
                pixel_size_m
            )
            
            if score_dict is None:
                continue
            
            score_dict.update({
                'Anomaly_ID': core_region.label,
                'Type': anomaly_type,
                'Centroid_Row': cy,
                'Centroid_Col': cx
            })
            
            results.append(score_dict)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        
        column_order = [
            'Anomaly_ID', 'Type', 'Centroid_Row', 'Centroid_Col',
            'IS', 'Severity', 'Area_m2', 'Area_pixels', 'Continuity',
            'Median_Delta_T', 'Mean_Boundary_Gradient', 'Residual_Std_Used', 'Raw_Score'
        ]
        
        return df[column_order]
    
    def _calculate_single_score(
        self,
        residual_2d: np.ndarray,
        full_anomaly_mask: np.ndarray,
        influence_only_mask: np.ndarray,
        residual_std: float,
        pixel_size_m: float
    ) -> Optional[Dict]:
        """
        Calculate Impact Score and submetrics for a single anomaly.
        
        Args:
            residual_2d: 2D array of LST residuals.
            full_anomaly_mask: Binary mask of full anomaly (core + influence).
            influence_only_mask: Binary mask of the influence zone only.
            residual_std: Standard deviation of residuals.
            pixel_size_m: Pixel size in meters.
            
        Returns:
            Dictionary containing IS and all submetrics, or None if invalid.
        """
        min_influence_pixels = self.params['min_influence_pixels']
        std_floor_degC = self.params['std_floor_degC']
        
        pixels_in_influence = residual_2d[influence_only_mask]
        n_pixels = pixels_in_influence.size

        if n_pixels < min_influence_pixels:
            return {
                'IS': 0.0,
                'Severity': 0.0,
                'Area_m2': 0.0,
                'Area_pixels': 0,
                'Continuity': 0.0,
                'Median_Delta_T': 0.0,
                'Mean_Boundary_Gradient': 0.0,
                'Residual_Std_Used': residual_std,
                'Raw_Score': 0.0
            }

        pixel_area_m2 = pixel_size_m ** 2
        area_m2 = n_pixels * pixel_area_m2
        
        median_delta_t = np.median(pixels_in_influence)
        
        sigma_temp_bg = np.maximum(residual_std, std_floor_degC)
        severity = np.abs(median_delta_t) / sigma_temp_bg

        mean_gradient = self._calculate_boundary_gradient(full_anomaly_mask)
        continuity = 1.0 / (1.0 + mean_gradient)

        raw_score = severity * area_m2 * continuity
        is_abs = np.log(1.0 + raw_score)
        is_signed = is_abs * np.sign(median_delta_t)

        if not np.isfinite(median_delta_t) or not np.isfinite(severity):
            return None
        
        return {
            'IS': is_signed,
            'Severity': severity,
            'Area_m2': area_m2,
            'Area_pixels': n_pixels,
            'Continuity': continuity,
            'Median_Delta_T': median_delta_t,
            'Mean_Boundary_Gradient': mean_gradient,
            'Residual_Std_Used': sigma_temp_bg,
            'Raw_Score': raw_score
        }
    
    def _calculate_boundary_gradient(self, anomaly_mask: np.ndarray) -> float:
        """
        Calculates the mean gradient magnitude at the heat anomaly boundary.
        
        Args:
            anomaly_mask: Binary mask of anomaly.
            
        Returns:
            Mean boundary gradient magnitude.
        """
        if self._gradient_magnitude_2d is None or self._gradient_magnitude_2d.size == 0:
            logger.warning("Gradient map not computed")
            return 0.0
            
        inner = find_boundaries(anomaly_mask, mode='inner', connectivity=1)
        outer = find_boundaries(anomaly_mask, mode='outer', connectivity=1)

        grad_inner = self._gradient_magnitude_2d[inner]
        grad_outer = self._gradient_magnitude_2d[outer]

        valid_gradients = np.concatenate([grad_inner, grad_outer])
        valid_gradients = valid_gradients[~np.isnan(valid_gradients)]

        # FIXED: Return 0.0 instead of interior mean when no boundary found
        if valid_gradients.size > 0:
            return np.mean(valid_gradients)
        else:
            logger.debug("No valid boundary gradients found, returning 0.0")
            return 0.0
