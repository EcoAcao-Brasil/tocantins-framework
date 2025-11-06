"""
Metrics calculation for the Tocantins Index.

Implements both Impact Score and Severity Score calculations.
Both metrics use standardized LST residuals as the measure of thermal anomalousness.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
from skimage import measure

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Unified calculator for Impact Score (IS) and Severity Score (SS).
    
    Impact Score: Quantifies the spatial extent and thermal significance of the Extended Anomaly Zone (EAZ)
    Severity Score: Quantifies thermal anomalousness of the core itself
    """
    
    DEFAULT_PARAMS = {
        'min_eaz_pixels': 1,
        'min_core_pixels': 1,
        'std_floor_degC': 0.05
    }
    
    def __init__(self, params: Dict = None):
        self.params = params or self.DEFAULT_PARAMS.copy()
        self._gradient_magnitude_2d = None
    
    def compute_gradient_map(self, residual_2d: np.ndarray) -> None:
        """Calculate spatial gradient magnitude of LST residuals."""
        grad_y, grad_x = np.gradient(np.nan_to_num(residual_2d))
        self._gradient_magnitude_2d = np.sqrt(grad_y**2 + grad_x**2)
    
    def calculate_impact_scores(
        self,
        hot_cores: np.ndarray,
        cold_cores: np.ndarray,
        hot_eaz: np.ndarray,
        cold_eaz: np.ndarray,
        residual_2d: np.ndarray,
        residual_std: float,
        pixel_size_m: float,
        connectivity: int = 2
    ) -> pd.DataFrame:
        """Calculate Impact Scores for all detected anomalies."""
        logger.info("Calculating Impact Scores")
        
        self.compute_gradient_map(residual_2d)
        
        hot_scores = self._score_impact(
            hot_cores, hot_eaz, 'hot',
            residual_2d, residual_std, pixel_size_m, connectivity
        )
        
        cold_scores = self._score_impact(
            cold_cores, cold_eaz, 'cold',
            residual_2d, residual_std, pixel_size_m, connectivity
        )
        
        all_scores = pd.concat([hot_scores, cold_scores], ignore_index=True)
        
        if not all_scores.empty:
            all_scores = all_scores.sort_values(by='IS', key=abs, ascending=False).reset_index(drop=True)
            logger.info(f"Calculated IS for {len(all_scores)} anomalies")
        else:
            logger.warning("No anomalies detected")
        
        return all_scores
    
    def calculate_severity_scores(
        self,
        hot_cores: np.ndarray,
        cold_cores: np.ndarray,
        residual_2d: np.ndarray,
        residual_std: float,
        pixel_size_m: float,
        connectivity: int = 2
    ) -> pd.DataFrame:
        """Calculate Severity Scores for all detected anomaly cores."""
        logger.info("Calculating Severity Scores")
        
        hot_scores = self._score_severity(
            hot_cores, 'hot', residual_2d, residual_std, pixel_size_m, connectivity
        )
        
        cold_scores = self._score_severity(
            cold_cores, 'cold', residual_2d, residual_std, pixel_size_m, connectivity
        )
        
        all_scores = pd.concat([hot_scores, cold_scores], ignore_index=True)
        
        if not all_scores.empty:
            all_scores = all_scores.sort_values(by='SS', key=abs, ascending=False).reset_index(drop=True)
            logger.info(f"Calculated SS for {len(all_scores)} anomaly cores")
        else:
            logger.warning("No anomaly cores detected")
        
        return all_scores
    
    def _score_impact(
        self,
        cores_mask: np.ndarray,
        eaz_mask: np.ndarray,
        anomaly_type: str,
        residual_2d: np.ndarray,
        residual_std: float,
        pixel_size_m: float,
        connectivity: int
    ) -> pd.DataFrame:
        """Score all anomalies of a given type for Impact Score."""
        if not np.any(cores_mask):
            return pd.DataFrame()
        
        labeled_cores = measure.label(cores_mask, connectivity=connectivity)
        full_anomalies_mask = cores_mask | eaz_mask
        labeled_full = measure.label(full_anomalies_mask, connectivity=connectivity)
        
        core_regions = measure.regionprops(labeled_cores)
        
        results = []
        for core_region in core_regions:
            cy, cx = core_region.centroid
            full_label = labeled_full[int(cy), int(cx)]
            
            if full_label == 0:
                continue
            
            full_mask = (labeled_full == full_label)
            eaz_only = full_mask & eaz_mask
            
            score_dict = self._calculate_impact_single(
                residual_2d, full_mask, eaz_only, residual_std, pixel_size_m
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
    
    def _score_severity(
        self,
        cores_mask: np.ndarray,
        anomaly_type: str,
        residual_2d: np.ndarray,
        residual_std: float,
        pixel_size_m: float,
        connectivity: int
    ) -> pd.DataFrame:
        """Score all cores of a given type for Severity Score."""
        if not np.any(cores_mask):
            return pd.DataFrame()
        
        labeled_cores = measure.label(cores_mask, connectivity=connectivity)
        core_regions = measure.regionprops(labeled_cores)
        
        results = []
        for core_region in core_regions:
            cy, cx = core_region.centroid
            core_label = labeled_cores[int(cy), int(cx)]
            
            if core_label == 0:
                continue
            
            core_mask = (labeled_cores == core_label)
            
            score_dict = self._calculate_severity_single(
                residual_2d, core_mask, residual_std, pixel_size_m
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
            'SS', 'Thermal_Intensity', 'Core_Area_m2', 'Core_Area_pixels',
            'Mean_Residual', 'Median_Residual', 'Residual_Std_Used', 'Raw_Score'
        ]
        
        return df[column_order]
    
    def _calculate_impact_single(
        self,
        residual_2d: np.ndarray,
        full_anomaly_mask: np.ndarray,
        eaz_only_mask: np.ndarray,
        residual_std: float,
        pixel_size_m: float
    ) -> Optional[Dict]:
        """
        Calculate Impact Score for a single anomaly.
        
        IS = sign(ΔT) × log(1 + severity × area × continuity)
        
        where:
            severity = |median(ΔT)| / σ_residual
            area = EAZ area (m²)
            continuity = 1 / (1 + mean_boundary_gradient)
        """
        min_pixels = self.params['min_eaz_pixels']
        std_floor = self.params['std_floor_degC']
        
        pixels_in_eaz = residual_2d[eaz_only_mask]
        n_pixels = pixels_in_eaz.size

        if n_pixels < min_pixels:
            return {
                'IS': 0.0, 'Severity': 0.0, 'Area_m2': 0.0, 'Area_pixels': 0,
                'Continuity': 0.0, 'Median_Delta_T': 0.0,
                'Mean_Boundary_Gradient': 0.0, 'Residual_Std_Used': residual_std,
                'Raw_Score': 0.0
            }

        pixel_area_m2 = pixel_size_m ** 2
        area_m2 = n_pixels * pixel_area_m2
        
        median_delta_t = np.median(pixels_in_eaz)
        
        sigma = np.maximum(residual_std, std_floor)
        severity = np.abs(median_delta_t) / sigma

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
            'Residual_Std_Used': sigma,
            'Raw_Score': raw_score
        }
    
    def _calculate_severity_single(
        self,
        residual_2d: np.ndarray,
        core_mask: np.ndarray,
        residual_std: float,
        pixel_size_m: float
    ) -> Optional[Dict]:
        """
        Calculate Severity Score for a single anomaly core.
        
        SS = sign(ΔT) × log(1 + thermal_intensity × area)
        
        where:
            thermal_intensity = |median(ΔT_core)| / σ_residual
            area = core area (m²)
        """
        min_pixels = self.params['min_core_pixels']
        std_floor = self.params['std_floor_degC']
        
        core_pixels = residual_2d[core_mask]
        n_pixels = core_pixels.size
        
        if n_pixels < min_pixels:
            return {
                'SS': 0.0, 'Thermal_Intensity': 0.0,
                'Core_Area_m2': 0.0, 'Core_Area_pixels': 0,
                'Mean_Residual': 0.0, 'Median_Residual': 0.0,
                'Residual_Std_Used': residual_std, 'Raw_Score': 0.0
            }
        
        pixel_area_m2 = pixel_size_m ** 2
        core_area_m2 = n_pixels * pixel_area_m2
        
        mean_residual = np.mean(core_pixels)
        median_residual = np.median(core_pixels)
        
        sigma = np.maximum(residual_std, std_floor)
        thermal_intensity = np.abs(median_residual) / sigma
        
        raw_score = thermal_intensity * core_area_m2
        ss_abs = np.log(1.0 + raw_score)
        ss_signed = ss_abs * np.sign(median_residual)
        
        if not np.isfinite(median_residual) or not np.isfinite(thermal_intensity):
            return None
        
        return {
            'SS': ss_signed,
            'Thermal_Intensity': thermal_intensity,
            'Core_Area_m2': core_area_m2,
            'Core_Area_pixels': n_pixels,
            'Mean_Residual': mean_residual,
            'Median_Residual': median_residual,
            'Residual_Std_Used': sigma,
            'Raw_Score': raw_score
        }
    
    def _calculate_boundary_gradient(self, anomaly_mask: np.ndarray) -> float:
        """Calculate mean gradient magnitude at anomaly boundary."""
        if self._gradient_magnitude_2d is None or self._gradient_magnitude_2d.size == 0:
            logger.warning("Gradient map not computed")
            return 0.0
            
        inner = find_boundaries(anomaly_mask, mode='inner', connectivity=1)
        outer = find_boundaries(anomaly_mask, mode='outer', connectivity=1)

        grad_inner = self._gradient_magnitude_2d[inner]
        grad_outer = self._gradient_magnitude_2d[outer]

        valid_gradients = np.concatenate([grad_inner, grad_outer])
        valid_gradients = valid_gradients[~np.isnan(valid_gradients)]

        if valid_gradients.size > 0:
            return np.mean(valid_gradients)
        else:
            logger.debug("No valid boundary gradients found")
            return 0.0
