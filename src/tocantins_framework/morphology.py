"""
Spatial Morphology Processing Module


This module implements morphological operations for spatial processing of
thermal anomaly masks, including core refinement and Extended Anomaly Zone
(EAZ) delineation.

Key Operations:
1. Core Unification
   - Morphological closing: Fill small gaps within anomalies
   - Agglutination: Merge nearby anomalies into unified cores
   - Size filtering: Remove artifacts below minimum size threshold

2. EAZ Growth
   - Spatial coherence: Grow zones from cores using connectivity
   - Threshold-based growth: Include adjacent anomalous pixels
   - Smoothing: Refine zone boundaries

Morphological Operations:
Binary Closing:
    Dilation followed by erosion. Fills small holes and gaps.
    Useful for connecting nearby anomaly pixels.

Binary Opening:
    Erosion followed by dilation. Removes small objects and smooths boundaries.
    Useful for cleaning noise and refining shapes.

Binary Dilation:
    Expands objects by adding pixels at boundaries.
    Controlled by structuring element (disk) size.

Binary Erosion:
    Shrinks objects by removing boundary pixels.
    Opposite of dilation.

Connectivity:
4-connectivity (connectivity=1):
    Pixels connected via edges (up, down, left, right).
    
8-connectivity (connectivity=2):
    Pixels connected via edges and corners (8 neighbors).
    Default for most operations.

Examples
--------
>>> from tocantins_framework.morphology import MorphologyProcessor
>>> 
>>> processor = MorphologyProcessor(params={
...     'min_anomaly_size': 5,
...     'agglutination_distance': 3
... })
>>> 
>>> # Unify anomaly cores
>>> unified_hot, unified_cold, hot_labels, cold_labels = \
...     processor.create_unified_cores(core_hot, core_cold)
>>> 
>>> # Grow Extended Anomaly Zones
>>> hot_eaz, cold_eaz = processor.grow_eaz(
...     unified_hot, unified_cold, residual_2d, 
...     valid_mask_2d, residual_std, k_threshold
... )
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np
from scipy import ndimage
from skimage import morphology, measure

logger = logging.getLogger(__name__)


class MorphologyProcessor:
    """
    Processor for spatial morphological operations on thermal anomaly masks.
    
    This class provides methods for refining anomaly cores through morphological
    operations and delineating Extended Anomaly Zones (EAZs) based on spatial
    coherence and thermal thresholds.
    
    Parameters
    ----------
    params : dict, optional
        Spatial processing parameters:
        - 'min_anomaly_size': Minimum pixels for valid anomaly (default: 1)
        - 'agglutination_distance': Dilation radius for merging (default: 4)
        - 'morphology_kernel_size': Kernel size for operations (default: 3)
        - 'connectivity': Pixel connectivity, 1 or 2 (default: 2)
    
    Attributes
    ----------
    params : dict
        Active parameter configuration.
    
    Methods
    -------
    create_unified_cores(core_hot, core_cold)
        Apply morphological operations to create unified anomaly cores.
    
    grow_eaz(hot_cores, cold_cores, residual_2d, valid_mask_2d, residual_std, k_threshold)
        Delineate Extended Anomaly Zones around anomaly cores.
    
    create_classification_map(shape, cold_eaz, hot_eaz, cold_cores, hot_cores)
        Generate multi-class classification map of all zones.
    
    Notes
    -----
    Morphological operations use disk-shaped structuring elements for
    isotropic (direction-independent) processing.
    
    The agglutination process merges nearby anomalies that likely represent
    parts of the same thermal phenomenon, improving spatial interpretability.
    
    Examples
    --------
    >>> processor = MorphologyProcessor()
    >>> unified_hot, unified_cold, _, _ = processor.create_unified_cores(
    ...     core_hot_mask, core_cold_mask
    ... )
    >>> print(f"Unified {np.sum(unified_hot)} hot pixels")
    """
    
    # Default spatial processing parameters
    DEFAULT_PARAMS = {
        'min_anomaly_size': 1,           # Minimum pixels for valid anomaly
        'agglutination_distance': 4,     # Pixels for core merging (dilation radius)
        'morphology_kernel_size': 3,     # Morphological operation kernel radius
        'connectivity': 2,               # 8-connectivity for connected components
    }
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize morphology processor.
        
        Parameters
        ----------
        params : dict, optional
            Custom spatial parameters (merges with defaults).
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        
        logger.debug(f"MorphologyProcessor initialized with params: {self.params}")
    
    def create_unified_cores(
        self,
        core_hot: np.ndarray,
        core_cold: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create unified anomaly cores using morphological operations.
        
        This method applies a sequence of morphological operations to:
        1. Fill small gaps within cores (closing)
        2. Merge nearby cores into unified regions (agglutination)
        3. Remove small isolated artifacts (size filtering)
        4. Label connected components for tracking
        
        The agglutination process is critical for identifying spatially
        coherent thermal anomalies that may be fragmented in the initial
        detection phase.
        
        Parameters
        ----------
        core_hot : np.ndarray
            2D boolean mask of hot anomaly cores.
        core_cold : np.ndarray
            2D boolean mask of cold anomaly cores.
        
        Returns
        -------
        unified_hot : np.ndarray
            2D boolean mask of unified hot cores.
        unified_cold : np.ndarray
            2D boolean mask of unified cold cores.
        hot_labels : np.ndarray
            2D integer array with unique labels for each hot core.
        cold_labels : np.ndarray
            2D integer array with unique labels for each cold core.
        
        Notes
        -----
        The morphological sequence is:
        1. Binary closing (kernel_size) - fill gaps
        2. Binary dilation (agglutination_distance) - expand for merging
        3. Binary opening (agglutination_distance) - smooth merged regions
        4. Size filtering (min_anomaly_size) - remove artifacts
        
        Examples
        --------
        >>> processor = MorphologyProcessor()
        >>> unified_hot, unified_cold, hot_ids, cold_ids = \
        ...     processor.create_unified_cores(core_hot, core_cold)
        >>> n_hot_cores = hot_ids.max()
        >>> n_cold_cores = cold_ids.max()
        >>> print(f"Identified {n_hot_cores} hot and {n_cold_cores} cold cores")
        """
        logger.info("Creating unified anomaly cores")
        
        # Create structuring elements
        kernel = morphology.disk(self.params['morphology_kernel_size'])
        agglut_kernel = morphology.disk(self.params['agglutination_distance'])
        
        # Process hot and cold cores
        unified_hot = self._process_cores(core_hot, kernel, agglut_kernel)
        unified_cold = self._process_cores(core_cold, kernel, agglut_kernel)
        
        # Label connected components
        connectivity = self.params['connectivity']
        hot_labels = measure.label(unified_hot, connectivity=connectivity)
        cold_labels = measure.label(unified_cold, connectivity=connectivity)
        
        # Log results
        n_hot = hot_labels.max()
        n_cold = cold_labels.max()
        pixels_hot = np.sum(unified_hot)
        pixels_cold = np.sum(unified_cold)
        
        logger.info(f"Created {n_hot} unified hot cores ({pixels_hot:,} pixels)")
        logger.info(f"Created {n_cold} unified cold cores ({pixels_cold:,} pixels)")
        
        return unified_hot, unified_cold, hot_labels, cold_labels
    
    def grow_eaz(
        self,
        hot_cores: np.ndarray,
        cold_cores: np.ndarray,
        residual_2d: np.ndarray,
        valid_mask_2d: np.ndarray,
        residual_std: float,
        k_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define spatially coherent Extended Anomaly Zones (EAZs) around cores.
        
        EAZs represent the full spatial extent of thermal anomalies, extending
        beyond immediate cores to include adjacent anomalous pixels. Zones are
        grown based on:
        1. Thermal criteria: Adjacent pixels exceeding residual threshold
        2. Spatial coherence: Connectivity to anomaly cores
        3. Boundary smoothing: Morphological refinement
        
        This approach captures the complete spatial impact of urban heat
        phenomena, including gradual transitions and indirect effects.
        
        Parameters
        ----------
        hot_cores : np.ndarray
            2D boolean mask of hot anomaly cores.
        cold_cores : np.ndarray
            2D boolean mask of cold anomaly cores.
        residual_2d : np.ndarray
            2D array of LST residuals (°C).
        valid_mask_2d : np.ndarray
            2D boolean mask of valid pixels.
        residual_std : float
            Standard deviation of training residuals (°C).
        k_threshold : float
            Threshold multiplier for residual-based detection.
        
        Returns
        -------
        hot_eaz : np.ndarray
            2D boolean mask of hot Extended Anomaly Zones (excluding cores).
        cold_eaz : np.ndarray
            2D boolean mask of cold Extended Anomaly Zones (excluding cores).
        
        Notes
        -----
        EAZ pixels are thermally anomalous (|residual| > k × σ) but were not
        included in the core due to not meeting the stricter M1 ∩ M2 criteria.
        
        EAZs are spatially connected to cores, ensuring they represent
        extensions of detected anomalies rather than isolated artifacts.
        
        Examples
        --------
        >>> hot_eaz, cold_eaz = processor.grow_eaz(
        ...     unified_hot, unified_cold, residual_2d, 
        ...     valid_mask_2d, residual_std=0.5, k_threshold=1.5
        ... )
        >>> print(f"Hot EAZ: {np.sum(hot_eaz):,} pixels")
        >>> print(f"Cold EAZ: {np.sum(cold_eaz):,} pixels")
        """
        logger.info("Growing Extended Anomaly Zones (EAZ)")
        
        # Calculate residual threshold
        threshold = k_threshold * residual_std
        logger.debug(f"EAZ residual threshold: ±{threshold:.4f}°C")
        
        # Identify potential EAZ pixels (anomalous but not in cores)
        potential_hot = (
            (residual_2d > threshold) & 
            valid_mask_2d & 
            ~hot_cores
        )
        potential_cold = (
            (residual_2d < -threshold) & 
            valid_mask_2d & 
            ~cold_cores
        )
        
        logger.debug(f"Potential hot EAZ pixels: {np.sum(potential_hot):,}")
        logger.debug(f"Potential cold EAZ pixels: {np.sum(potential_cold):,}")
        
        # Grow zones from cores
        hot_eaz = self._grow_zone(hot_cores, potential_hot)
        cold_eaz = self._grow_zone(cold_cores, potential_cold)
        
        # Log final statistics
        n_hot_eaz = np.sum(hot_eaz)
        n_cold_eaz = np.sum(cold_eaz)
        n_hot_core = np.sum(hot_cores)
        n_cold_core = np.sum(cold_cores)
        
        logger.info(f"Hot anomaly: {n_hot_core:,} core + {n_hot_eaz:,} EAZ pixels")
        logger.info(f"Cold anomaly: {n_cold_core:,} core + {n_cold_eaz:,} EAZ pixels")
        
        return hot_eaz, cold_eaz
    
    def create_classification_map(
        self,
        shape: Tuple[int, int],
        cold_eaz: np.ndarray,
        hot_eaz: np.ndarray,
        cold_cores: np.ndarray,
        hot_cores: np.ndarray
    ) -> np.ndarray:
        """
        Create multi-class classification map with all thermal zones.
        
        Generates a single raster with distinct classes for background,
        cold EAZ, hot EAZ, cold cores, and hot cores. Useful for visualization
        and spatial analysis.
        
        Parameters
        ----------
        shape : tuple of int
            Output array shape (height, width).
        cold_eaz : np.ndarray
            2D boolean mask of cold Extended Anomaly Zones.
        hot_eaz : np.ndarray
            2D boolean mask of hot Extended Anomaly Zones.
        cold_cores : np.ndarray
            2D boolean mask of cold anomaly cores.
        hot_cores : np.ndarray
            2D boolean mask of hot anomaly cores.
        
        Returns
        -------
        classification : np.ndarray
            2D uint8 array with class values:
            - 0: Background (normal/non-anomalous)
            - 1: Cold Extended Anomaly Zone
            - 2: Hot Extended Anomaly Zone
            - 3: Cold anomaly core
            - 4: Hot anomaly core
        
        Notes
        -----
        Classes are assigned in hierarchical order (background → EAZ → core),
        so cores overwrite EAZ pixels if they overlap.
        
        Examples
        --------
        >>> classification = processor.create_classification_map(
        ...     (5000, 5000), cold_eaz, hot_eaz, cold_cores, hot_cores
        ... )
        >>> unique, counts = np.unique(classification, return_counts=True)
        >>> for cls, count in zip(unique, counts):
        ...     print(f"Class {cls}: {count:,} pixels")
        """
        logger.info("Creating classification map")
        
        # Initialize with background (0)
        classification = np.zeros(shape, dtype=np.uint8)
        
        # Assign classes hierarchically
        classification[cold_eaz] = 1    # Cold EAZ
        classification[hot_eaz] = 2     # Hot EAZ
        classification[cold_cores] = 3  # Cold core (overwrites EAZ)
        classification[hot_cores] = 4   # Hot core (overwrites EAZ)
        
        # Log class distribution
        unique, counts = np.unique(classification, return_counts=True)
        class_names = {
            0: 'Background',
            1: 'Cold EAZ',
            2: 'Hot EAZ',
            3: 'Cold Core',
            4: 'Hot Core'
        }
        
        logger.debug("Classification map distribution:")
        for cls, count in zip(unique, counts):
            pct = 100 * count / classification.size
            logger.debug(f"  {class_names.get(cls, f'Class {cls}')}: "
                        f"{count:,} pixels ({pct:.2f}%)")
        
        return classification
    
    def _process_cores(
        self,
        core_mask: np.ndarray,
        kernel: np.ndarray,
        agglut_kernel: np.ndarray
    ) -> np.ndarray:
        """
        Apply morphological operation sequence to refine anomaly cores.
        
        Parameters
        ----------
        core_mask : np.ndarray
            Initial binary mask of anomaly cores.
        kernel : np.ndarray
            Structuring element for closing operation.
        agglut_kernel : np.ndarray
            Larger structuring element for agglutination.
        
        Returns
        -------
        processed : np.ndarray
            Processed binary mask of unified cores.
        
        Notes
        -----
        Morphological sequence:
        1. Binary closing: Fill small gaps
        2. Binary dilation: Expand for merging
        3. Binary opening: Smooth merged regions
        4. Size filtering: Remove small artifacts
        """
        if not np.any(core_mask):
            logger.debug("Empty core mask, skipping processing")
            return core_mask
        
        # Step 1: Close small gaps
        closed = morphology.binary_closing(core_mask, kernel)
        
        # Step 2: Dilate for agglutination
        dilated = morphology.binary_dilation(closed, agglut_kernel)
        
        # Step 3: Open to smooth
        opened = morphology.binary_opening(dilated, agglut_kernel)
        
        # Step 4: Remove small objects
        processed = morphology.remove_small_objects(
            opened, 
            min_size=self.params['min_anomaly_size']
        )
        
        return processed
    
    def _grow_zone(
        self,
        cores: np.ndarray,
        potential_eaz: np.ndarray
    ) -> np.ndarray:
        """
        Grow an Extended Anomaly Zone from cores using spatial connectivity.
        
        This method ensures EAZ pixels are spatially connected to cores,
        preventing isolated anomalous pixels from being misclassified as
        part of the EAZ.
        
        Parameters
        ----------
        cores : np.ndarray
            2D boolean mask of anomaly cores (seeds for growth).
        potential_eaz : np.ndarray
            2D boolean mask of potential EAZ pixels (growth candidates).
        
        Returns
        -------
        eaz : np.ndarray
            2D boolean mask of Extended Anomaly Zone (excluding cores).
        
        Notes
        -----
        Growth process:
        1. Combine cores and potential EAZ pixels
        2. Label connected components
        3. Keep only components containing core pixels
        4. Subtract cores to get EAZ-only mask
        5. Apply morphological smoothing
        
        Examples
        --------
        >>> eaz = processor._grow_zone(core_mask, potential_pixels)
        >>> print(f"Grew zone to {np.sum(eaz)} pixels")
        """
        if not np.any(cores):
            logger.debug("No cores provided, returning empty EAZ")
            return np.zeros_like(cores, dtype=bool)
        
        # Combine cores and potential EAZ
        all_anomalies = cores | potential_eaz
        
        # Label connected components
        labeled_zones, n_labels = ndimage.label(all_anomalies)
        
        # Identify labels that contain core pixels
        core_blob_labels = np.unique(labeled_zones[cores])
        core_blob_labels = core_blob_labels[core_blob_labels > 0]  # Remove background
        
        # Keep only connected components containing cores
        connected_mask = np.isin(labeled_zones, core_blob_labels)
        
        # EAZ = connected components minus cores
        eaz = connected_mask & ~cores
        
        # Smooth EAZ boundaries
        smoothing_kernel = morphology.disk(1)
        eaz = morphology.binary_closing(eaz, smoothing_kernel)
        eaz = morphology.binary_opening(eaz, smoothing_kernel)
        
        return eaz