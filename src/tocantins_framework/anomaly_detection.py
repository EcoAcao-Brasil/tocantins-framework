"""
Thermal Anomaly Detection Module

This module implements a two-stage anomaly detection methodology combining
statistical thresholding with machine learning-based residual analysis.

Methodology:
Stage 1: Statistical Detection (M1)
    Uses percentile-based thresholds to identify initial anomaly candidates:
    - Cold anomalies: LST ≤ 2nd percentile
    - Hot anomalies: LST ≥ 98th percentile

Stage 2: ML-Based Refinement (M2)
    Random Forest regression predicts expected LST from spectral indices,
    then residual analysis identifies pixels with anomalous thermal behavior:
    - Residual = Observed LST - Predicted LST
    - Core anomalies: |Residual| > k × σ_residual (typically k=1.5)

Core Anomalies
    Final cores are the intersection of M1 and M2 (M1 ∩ M2), ensuring both
    statistical and contextual anomalousness.

The Random Forest model is trained only on non-anomalous pixels (M1 negative)
to learn the normal relationship between spectral indices and LST.

Examples:
>>> from tocantins_framework.anomaly_detection import AnomalyDetector
>>> import pandas as pd
>>> import numpy as np
>>> 
>>> # Initialize detector
>>> detector = AnomalyDetector(k_threshold=1.5)
>>> 
>>> # Stage 1: Statistical detection
>>> m1_hot, m1_cold, data = detector.detect_statistical_anomalies(
...     data, lst_2d, valid_mask_2d
... )
>>> 
>>> # Stage 2: Train model and refine cores
>>> detector.train_model(data)
>>> residual_2d, data = detector.calculate_residuals(data, lst_2d)
>>> core_hot, core_cold = detector.refine_anomaly_cores(
...     m1_hot, m1_cold, residual_2d, valid_mask_2d
... )
"""

import logging
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Two-stage thermal anomaly detector using statistical and ML-based methods.
    
    This class implements a robust anomaly detection pipeline that combines
    percentile-based statistical thresholds with Random Forest regression
    residual analysis for accurate thermal anomaly identification.
    
    Parameters
    ----------
    k_threshold : float, default=1.5
        Threshold multiplier for residual-based detection.
        Anomalies defined as |residual| > k_threshold × σ_residual.
        Typical values: 1.0-2.0 (higher = more conservative).
    
    rf_params : dict, optional
        Random Forest hyperparameters. If None, uses DEFAULT_RF_PARAMS.
        See sklearn.ensemble.RandomForestRegressor for parameter details.
    
    Attributes
    ----------
    rf_model : RandomForestRegressor or None
        Trained Random Forest model (None before training).
    
    training_stats : dict
        Training statistics including:
        - 'r2': Model R² score on training data
        - 'residual_std': Standard deviation of training residuals
        - 'rmse': Root mean squared error
        - 'n_training_samples': Number of training samples
    
    Methods
    -------
    detect_statistical_anomalies(data, lst_2d, valid_mask_2d)
        Stage 1: Percentile-based anomaly detection.
    
    train_model(data)
        Train Random Forest on non-anomalous pixels.
    
    calculate_residuals(data, lst_2d)
        Calculate LST residuals from RF predictions.
    
    refine_anomaly_cores(m1_hot, m1_cold, residual_2d, valid_mask_2d)
        Stage 2: Refine cores using residual thresholds.
    
    get_training_stats()
        Retrieve training statistics dictionary.
    
    Notes
    -----
    The detector is designed to be robust to varying urban contexts by learning
    the normal LST-spectral relationship from the scene itself, rather than
    using fixed global thresholds.
    
    Examples
    --------
    >>> detector = AnomalyDetector(k_threshold=1.5)
    >>> m1_hot, m1_cold, data = detector.detect_statistical_anomalies(
    ...     data, lst_2d, valid_mask_2d
    ... )
    >>> model = detector.train_model(data)
    >>> print(f"Model R²: {detector.training_stats['r2']:.3f}")
    """
    
    # Default Random Forest hyperparameters optimized for LST prediction
    DEFAULT_RF_PARAMS = {
        'n_estimators': 200,        # Number of trees in forest
        'max_depth': 25,            # Maximum tree depth
        'min_samples_split': 8,     # Minimum samples to split node
        'min_samples_leaf': 4,      # Minimum samples per leaf
        'max_features': 'sqrt',     # Features considered per split
        'random_state': 42,         # Reproducibility seed
        'n_jobs': -1,               # Use all CPU cores
        'bootstrap': True,          # Bootstrap sampling
        'oob_score': False,         # Don't compute OOB score (faster)
    }
    
    # Percentile thresholds for statistical anomaly detection
    COLD_PERCENTILE = 2   # 2nd percentile for cold anomalies
    HOT_PERCENTILE = 98   # 98th percentile for hot anomalies
    
    def __init__(self, k_threshold: float = 1.5, rf_params: Optional[Dict] = None):
        """
        Initialize thermal anomaly detector.
        
        Parameters
        ----------
        k_threshold : float, default=1.5
            Residual threshold multiplier.
        rf_params : dict, optional
            Random Forest parameters (uses defaults if None).
        """
        self.k_threshold = k_threshold
        self.rf_params = rf_params or self.DEFAULT_RF_PARAMS.copy()
        self.rf_model = None
        self.training_stats = {}
        
        logger.debug(f"AnomalyDetector initialized with k={k_threshold}")
    
    def detect_statistical_anomalies(
        self,
        data: pd.DataFrame,
        lst_2d: np.ndarray,
        valid_mask_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Detect statistical anomalies using percentile thresholds (Stage 1: M1).
        
        Identifies pixels with LST values in the extreme tails of the
        distribution (2nd and 98th percentiles), representing statistically
        unusual thermal conditions.
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing 'LST' column with temperature values.
        lst_2d : np.ndarray
            2D Land Surface Temperature array.
        valid_mask_2d : np.ndarray
            2D boolean mask of valid pixels.
        
        Returns
        -------
        m1_hot_2d : np.ndarray
            2D boolean mask of hot anomalies (LST ≥ 98th percentile).
        m1_cold_2d : np.ndarray
            2D boolean mask of cold anomalies (LST ≤ 2nd percentile).
        data : pd.DataFrame
            Updated DataFrame with 'M1_anomaly' boolean column.
        
        Notes
        -----
        The 2nd and 98th percentiles are used to capture approximately 4% of
        pixels as potential anomalies, balancing sensitivity and specificity.
        
        Examples
        --------
        >>> m1_hot, m1_cold, data = detector.detect_statistical_anomalies(
        ...     data, lst_2d, valid_mask_2d
        ... )
        >>> print(f"Hot anomalies: {np.sum(m1_hot):,} pixels")
        >>> print(f"Cold anomalies: {np.sum(m1_cold):,} pixels")
        """
        logger.info("Stage 1: Statistical anomaly detection (M1)")
        
        # Calculate percentile thresholds
        p_cold = np.percentile(data['LST'], self.COLD_PERCENTILE)
        p_hot = np.percentile(data['LST'], self.HOT_PERCENTILE)
        
        logger.debug(f"Cold threshold (P{self.COLD_PERCENTILE}): {p_cold:.2f}°C")
        logger.debug(f"Hot threshold (P{self.HOT_PERCENTILE}): {p_hot:.2f}°C")
        
        # Create 2D masks
        m1_cold_2d = (lst_2d <= p_cold) & valid_mask_2d
        m1_hot_2d = (lst_2d >= p_hot) & valid_mask_2d
        
        # Add anomaly flag to DataFrame
        data = data.copy()
        data['M1_anomaly'] = (data['LST'] <= p_cold) | (data['LST'] >= p_hot)
        
        # Log statistics
        n_cold = np.sum(m1_cold_2d)
        n_hot = np.sum(m1_hot_2d)
        n_total = np.sum(valid_mask_2d)
        pct_anomalies = 100 * (n_cold + n_hot) / n_total
        
        logger.info(f"M1 detected {n_hot:,} hot and {n_cold:,} cold anomalies")
        logger.info(f"Total M1 anomalies: {pct_anomalies:.2f}% of valid pixels")
        
        return m1_hot_2d, m1_cold_2d, data
    
    def train_model(self, data: pd.DataFrame) -> RandomForestRegressor:
        """
        Train Random Forest model on non-anomalous pixels.
        
        Learns the normal relationship between spectral indices (NDVI, NDWI,
        NDBI, NDBSI) and LST using only pixels that were not flagged as
        statistical anomalies (M1 negative).
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing columns:
            - 'M1_anomaly': Boolean flag from statistical detection
            - 'NDVI', 'NDWI', 'NDBI', 'NDBSI': Spectral indices
            - 'LST': Land Surface Temperature
        
        Returns
        -------
        rf_model : RandomForestRegressor
            Trained Random Forest model.
        
        Notes
        -----
        Training statistics are stored in self.training_stats and can be
        accessed via get_training_stats().
        
        The model uses only non-anomalous pixels to avoid learning anomalous
        patterns as normal behavior.
        
        Examples
        --------
        >>> model = detector.train_model(data)
        >>> stats = detector.get_training_stats()
        >>> print(f"R² = {stats['r2']:.3f}, σ = {stats['residual_std']:.3f}°C")
        """
        logger.info("Stage 2: Training Random Forest model")
        
        # Filter training data (non-anomalous pixels only)
        training_data = data[~data['M1_anomaly']].copy()
        n_training = len(training_data)
        
        if n_training == 0:
            raise ValueError("No non-anomalous pixels available for training")
        
        # Prepare features and target
        X_train = training_data[['NDVI', 'NDWI', 'NDBI', 'NDBSI']].values
        y_train = training_data['LST'].values
        
        logger.debug(f"Training on {n_training:,} non-anomalous pixels")
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(**self.rf_params)
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate model performance
        y_pred_train = self.rf_model.predict(X_train)
        r2 = r2_score(y_train, y_pred_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        residuals = y_train - y_pred_train
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # Store statistics
        self.training_stats = {
            'r2': r2,
            'rmse': rmse,
            'residual_std': residual_std,
            'residual_mean': residual_mean,
            'n_training_samples': n_training,
            'threshold': self.k_threshold * residual_std
        }
        
        # Log performance metrics
        logger.info(f"Model training complete:")
        logger.info(f"  R² score: {r2:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}°C")
        logger.info(f"  Residual σ: {residual_std:.4f}°C (mean: {residual_mean:.4f}°C)")
        logger.info(f"  Anomaly threshold: ±{self.training_stats['threshold']:.4f}°C")
        
        # Feature importance analysis
        feature_names = ['NDVI', 'NDWI', 'NDBI', 'NDBSI']
        importances = self.rf_model.feature_importances_
        logger.debug("Feature importances:")
        for name, importance in zip(feature_names, importances):
            logger.debug(f"  {name}: {importance:.4f}")
        
        return self.rf_model
    
    def calculate_residuals(
        self,
        data: pd.DataFrame,
        lst_2d: np.ndarray
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Calculate LST residuals from Random Forest predictions.
        
        Residuals represent the difference between observed and expected LST:
        Residual = LST_observed - LST_predicted
        
        Positive residuals indicate warmer-than-expected pixels (hot anomalies),
        while negative residuals indicate cooler-than-expected pixels (cold anomalies).
        
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with spectral indices (NDVI, NDWI, NDBI, NDBSI).
        lst_2d : np.ndarray
            2D LST array for spatial mapping of residuals.
        
        Returns
        -------
        residual_2d : np.ndarray
            2D array of LST residuals with same shape as lst_2d.
        data : pd.DataFrame
            Updated DataFrame with added columns:
            - 'LST_predicted': Predicted LST from RF model
            - 'LST_residual': Observed - Predicted LST
        
        Raises
        ------
        RuntimeError
            If train_model() has not been called yet.
        
        Examples
        --------
        >>> residual_2d, data = detector.calculate_residuals(data, lst_2d)
        >>> print(f"Residual range: {data['LST_residual'].min():.2f} to "
        ...       f"{data['LST_residual'].max():.2f}°C")
        """
        if self.rf_model is None:
            raise RuntimeError("Model not trained. Call train_model() first.")
        
        logger.info("Calculating LST residuals")
        
        # Predict LST for all pixels
        X_all = data[['NDVI', 'NDWI', 'NDBI', 'NDBSI']].values
        
        data = data.copy()
        data['LST_predicted'] = self.rf_model.predict(X_all)
        data['LST_residual'] = data['LST'] - data['LST_predicted']
        
        # Create 2D residual map
        residual_2d = np.full(lst_2d.shape, np.nan)
        rows = data['row'].astype(int).values
        cols = data['col'].astype(int).values
        residual_2d[rows, cols] = data['LST_residual'].values
        
        # Log residual statistics
        residuals = data['LST_residual'].values
        logger.debug(f"Residual statistics:")
        logger.debug(f"  Mean: {np.mean(residuals):.4f}°C")
        logger.debug(f"  Std: {np.std(residuals):.4f}°C")
        logger.debug(f"  Range: [{np.min(residuals):.2f}, {np.max(residuals):.2f}]°C")
        
        return residual_2d, data
    
    def refine_anomaly_cores(
        self,
        m1_hot_2d: np.ndarray,
        m1_cold_2d: np.ndarray,
        residual_2d: np.ndarray,
        valid_mask_2d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine anomaly cores using residual-based detection (Stage 2: M2).
        
        Final cores are the intersection of statistical anomalies (M1) and
        residual-based anomalies (M2), ensuring both statistical and contextual
        anomalousness:
        
        Core_hot = M1_hot ∩ M2_hot
        Core_cold = M1_cold ∩ M2_cold
        
        where M2 is defined by |residual| > k × σ_residual.
        
        Parameters
        ----------
        m1_hot_2d : np.ndarray
            Statistical hot anomaly mask (M1).
        m1_cold_2d : np.ndarray
            Statistical cold anomaly mask (M1).
        residual_2d : np.ndarray
            LST residual array.
        valid_mask_2d : np.ndarray
            Valid pixel mask.
        
        Returns
        -------
        core_hot_2d : np.ndarray
            Refined hot anomaly cores (M1 ∩ M2).
        core_cold_2d : np.ndarray
            Refined cold anomaly cores (M1 ∩ M2).
        
        Notes
        -----
        This two-stage approach reduces false positives by requiring anomalies
        to be both statistically extreme AND unexplained by normal spectral-
        thermal relationships.
        
        Examples
        --------
        >>> core_hot, core_cold = detector.refine_anomaly_cores(
        ...     m1_hot, m1_cold, residual_2d, valid_mask_2d
        ... )
        >>> print(f"Refined to {np.sum(core_hot):,} hot cores")
        >>> print(f"Refined to {np.sum(core_cold):,} cold cores")
        """
        logger.info("Stage 2: Refining anomaly cores using residuals (M2)")
        
        # Calculate residual threshold
        threshold = self.k_threshold * self.training_stats['residual_std']
        logger.debug(f"Residual threshold: ±{threshold:.4f}°C")
        
        # Residual-based anomaly masks (M2)
        m2_hot_2d = (residual_2d > threshold) & valid_mask_2d
        m2_cold_2d = (residual_2d < -threshold) & valid_mask_2d
        
        # Intersection: M1 ∩ M2
        core_hot_2d = m1_hot_2d & m2_hot_2d
        core_cold_2d = m1_cold_2d & m2_cold_2d
        
        # Log refinement statistics
        n_m1_hot = np.sum(m1_hot_2d)
        n_m2_hot = np.sum(m2_hot_2d)
        n_core_hot = np.sum(core_hot_2d)
        
        n_m1_cold = np.sum(m1_cold_2d)
        n_m2_cold = np.sum(m2_cold_2d)
        n_core_cold = np.sum(core_cold_2d)
        
        logger.info(f"Hot anomalies: {n_m1_hot:,} (M1) → {n_core_hot:,} (M1∩M2)")
        logger.info(f"Cold anomalies: {n_m1_cold:,} (M1) → {n_core_cold:,} (M1∩M2)")
        
        if n_m1_hot > 0:
            pct_hot_retained = 100 * n_core_hot / n_m1_hot
            logger.debug(f"Hot core retention: {pct_hot_retained:.1f}%")
        
        if n_m1_cold > 0:
            pct_cold_retained = 100 * n_core_cold / n_m1_cold
            logger.debug(f"Cold core retention: {pct_cold_retained:.1f}%")
        
        return core_hot_2d, core_cold_2d
    
    def get_training_stats(self) -> Dict:
        """
        Get training statistics dictionary.
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - 'r2': Model R² score
            - 'rmse': Root mean squared error (°C)
            - 'residual_std': Residual standard deviation (°C)
            - 'residual_mean': Mean residual (°C)
            - 'n_training_samples': Number of training pixels
            - 'threshold': Anomaly detection threshold (k × σ)
        
        Examples
        --------
        >>> stats = detector.get_training_stats()
        >>> print(f"Model quality: R²={stats['r2']:.3f}, RMSE={stats['rmse']:.2f}°C")
        """
        return self.training_stats.copy()