"""
Anomaly detection module.

Implements statistical and machine learning-based methods of anomaly detection.
"""

import logging
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Thermal anomaly detector using statistical thresholds and Random Forest regression.
    """
    
    DEFAULT_RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': 25,
        'min_samples_split': 8,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1
    }
    
    def __init__(self, k_threshold: float = 1.5, rf_params: Dict = None):
        """
        Initialize anomaly detector.
        
        Args:
            k_threshold: Threshold multiplier for residual-based detection.
            rf_params: Random Forest model parameters.
        """
        self.k_threshold = k_threshold
        self.rf_params = rf_params or self.DEFAULT_RF_PARAMS.copy()
        self.rf_model = None
        self.training_stats = {}
    
    def detect_statistical_anomalies(
        self,
        data: pd.DataFrame,
        lst_2d: np.ndarray,
        valid_mask_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Detect statistical anomalies using percentile thresholds.
        
        Args:
            data: DataFrame with LST values.
            lst_2d: 2D LST array.
            valid_mask_2d: 2D valid pixel mask.
            
        Returns:
            Tuple of (hot anomaly mask, cold anomaly mask, updated DataFrame).
        """
        logger.info("Detecting statistical anomalies")
        
        p2 = np.percentile(data['LST'], 2)
        p98 = np.percentile(data['LST'], 98)
        
        m1_cold_2d = (lst_2d <= p2) & valid_mask_2d
        m1_hot_2d = (lst_2d >= p98) & valid_mask_2d
        
        data = data.copy()
        data['M1_anomaly'] = (data['LST'] <= p2) | (data['LST'] >= p98)
        
        n_anomalies = np.sum(m1_hot_2d | m1_cold_2d)
        logger.info(f"Detected {n_anomalies:,} statistical anomalies")
        
        return m1_hot_2d, m1_cold_2d, data
    
    def train_model(self, data: pd.DataFrame) -> RandomForestRegressor:
        """
        Trains a Random Forest model on non-anomalous pixels.
        
        Args:
            data: DataFrame with spectral indices and anomaly labels.
            
        Returns:
            Trained Random Forest model.
        """
        logger.info("Training Random Forest model")
        
        training_data = data[~data['M1_anomaly']].copy()
        X_train = training_data[['NDVI', 'NDWI', 'NDBI', 'NDBSI']].values
        y_train = training_data['LST'].values
        
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.rf_model.fit(X_train, y_train)
        
        y_pred_train = self.rf_model.predict(X_train)
        r2 = r2_score(y_train, y_pred_train)
        residual_std = np.std(y_train - y_pred_train)
        
        self.training_stats = {'r2': r2, 'residual_std': residual_std}
        
        logger.info(f"Model trained (R² = {r2:.4f}, σ = {residual_std:.4f}°C)")
        
        return self.rf_model
    
    def calculate_residuals(
        self,
        data: pd.DataFrame,
        lst_2d: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate LST residuals from Random Forest predictions.
        
        Args:
            data: DataFrame with spectral indices.
            lst_2d: 2D LST array for spatial mapping.
            
        Returns:
            Tuple of (2D residual array, updated DataFrame).
        """
        logger.info("Calculating LST residuals")
        
        X_all = data[['NDVI', 'NDWI', 'NDBI', 'NDBSI']].values
        
        data = data.copy()
        data['LST_predicted'] = self.rf_model.predict(X_all)
        data['LST_residual'] = data['LST'] - data['LST_predicted']
        
        residual_2d = np.full(lst_2d.shape, np.nan)
        rows = data['row'].astype(int)
        cols = data['col'].astype(int)
        residual_2d[rows, cols] = data['LST_residual']
        
        return residual_2d, data
    
    def refine_anomaly_cores(
        self,
        m1_hot_2d: np.ndarray,
        m1_cold_2d: np.ndarray,
        residual_2d: np.ndarray,
        valid_mask_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine anomaly cores using residual-based detection.
        
        Args:
            m1_hot_2d: Statistical hot anomaly mask.
            m1_cold_2d: Statistical cold anomaly mask.
            residual_2d: LST residual array.
            valid_mask_2d: Valid pixel mask.
            
        Returns:
            Tuple of (refined hot cores, refined cold cores).
        """
        threshold = self.k_threshold * self.training_stats['residual_std']
        
        m2_hot_2d = (residual_2d > threshold) & valid_mask_2d
        m2_cold_2d = (residual_2d < -threshold) & valid_mask_2d
        
        core_hot_2d = m1_hot_2d & m2_hot_2d
        core_cold_2d = m1_cold_2d & m2_cold_2d
        
        return core_hot_2d, core_cold_2d
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return self.training_stats
