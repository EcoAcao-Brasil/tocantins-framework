"""
Morphological operations for spatial anomaly processing.

Handles anomaly core refinement and defines Extended Anomaly Zones (EAZs).
"""

import logging
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage
from skimage import morphology, measure

logger = logging.getLogger(__name__)


class MorphologyProcessor:
    """
    Processor for spatial morphological operations on anomaly masks.
    """
    
    DEFAULT_PARAMS = {
        'min_anomaly_size': 1,
        'agglutination_distance': 4,
        'morphology_kernel_size': 3,
        'connectivity': 2
    }
    
    def __init__(self, params: Dict = None):
        """
        Initializes the morphological operations processor.
        
        Args:
            params: Spatial processing parameters.
        """
        self.params = params or self.DEFAULT_PARAMS.copy()
    
    def create_unified_cores(
        self,
        core_hot: np.ndarray,
        core_cold: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create unified anomaly cores using morphological operations.
        
        Args:
            core_hot: Initial hot anomaly core mask.
            core_cold: Initial cold anomaly core mask.
            
        Returns:
            Tuple of (unified hot cores, unified cold cores, 
                     hot core labels, cold core labels).
        """
        logger.info("Creating unified anomaly cores")
        
        kernel = morphology.disk(self.params['morphology_kernel_size'])
        agglut_kernel = morphology.disk(self.params['agglutination_distance'])
        
        unified_hot = self._process_cores(core_hot, kernel, agglut_kernel)
        unified_cold = self._process_cores(core_cold, kernel, agglut_kernel)
        
        connectivity = self.params['connectivity']
        hot_labels = measure.label(unified_hot, connectivity=connectivity)
        cold_labels = measure.label(unified_cold, connectivity=connectivity)
        
        logger.info(f"Created {hot_labels.max()} hot and {cold_labels.max()} cold cores")
        
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
        Define spatially coherent Extended Anomaly Zones around anomaly cores.
        
        Args:
            hot_cores: Hot anomaly core mask.
            cold_cores: Cold anomaly core mask.
            residual_2d: LST residual array.
            valid_mask_2d: Valid pixel mask.
            residual_std: Standard deviation of residuals.
            k_threshold: Threshold multiplier.
            
        Returns:
            Tuple of (hot EAZs zone mask, cold EAZs zone mask).
        """
        logger.info("Growing Extended Anomaly Zones")
        
        threshold = k_threshold * residual_std
        
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
        
        hot_eaz = self._grow_zone(hot_cores, potential_hot)
        cold_eaz = self._grow_zone(cold_cores, potential_cold)
        
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
        Creates a final classification map with all zones.
        
        Args:
            shape: Output array shape.
            cold_eaz: Cold EAZs zone mask.
            hot_eaz: Hot EAZs zone mask.
            cold_cores: Cold anomaly core mask.
            hot_cores: Hot anomaly core mask.
            
        Returns:
            Classification map (0=background, 1=cold eaz, 2=hot eaz,
                              3=cold core, 4=hot core).
        """
        logger.info("Creating classification map")
        
        classification = np.zeros(shape, dtype=np.uint8)
        classification[cold_eaz] = 1
        classification[hot_eaz] = 2
        classification[cold_cores] = 3
        classification[hot_cores] = 4
        
        return classification
    
    def _process_cores(
        self,
        core_mask: np.ndarray,
        kernel: np.ndarray,
        agglut_kernel: np.ndarray
    ) -> np.ndarray:
        """
        Apply morphological operations to refine anomaly cores.
        
        Args:
            core_mask: Initial binary mask of anomaly cores.
            kernel: Morphological structuring element.
            agglut_kernel: Agglutination structuring element.
            
        Returns:
            Processed binary mask of unified cores.
        """
        closed = morphology.binary_closing(core_mask, kernel)
        dilated = morphology.binary_dilation(closed, agglut_kernel)
        opened = morphology.binary_opening(dilated, agglut_kernel)
        
        return morphology.remove_small_objects(
            opened, 
            min_size=self.params['min_anomaly_size']
        )
    
    def _grow_zone(
        self,
        cores: np.ndarray,
        potential_eaz: np.ndarray
    ) -> np.ndarray:
        """
        Grow an EAZ from anomaly cores.
        
        Args:
            cores: Binary mask of anomaly cores.
            potential_eaz: Binary mask of potential EAZ pixels.
            
        Returns:
            Binary mask of the EAZs.
        """
        if not np.any(cores):
            return np.zeros_like(cores, dtype=bool)
        
        all_anomalies = cores | potential_eaz
        labeled_zones, _ = ndimage.label(all_anomalies)
        core_blob_labels = np.unique(labeled_zones[cores])
        connected_mask = np.isin(labeled_zones, core_blob_labels)
        eaz = connected_mask & ~cores
        
        smoothing_kernel = morphology.disk(1)
        eaz = morphology.binary_closing(eaz, smoothing_kernel)
        eaz = morphology.binary_opening(eaz, smoothing_kernel)
        
        return eaz
