"""
Input/Output operations for saving analysis results.

Handles CSV exports and GeoTIFF writing.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import rasterio

logger = logging.getLogger(__name__)


class ResultsWriter:
    """
    Writer for analysis results, including CSV and GeoTIFF outputs.
    """
    
    def __init__(self, raster_meta: Dict):
        """
        Initialize results writer.
        
        Args:
            raster_meta: Metadata from source raster.
        """
        self.raster_meta = raster_meta
    
    def save_all(
        self,
        output_dir: str,
        impact_scores: pd.DataFrame,
        classification_map: np.ndarray,
        residual_map: np.ndarray
    ) -> None:
        """
        Saves all analysis results to the output directory.
        
        Args:
            output_dir: Output directory path.
            impact_scores: Dataframe with all Impact Scores for each heat anomaly.
            classification_map: Classification map array.
            residual_map: LST residual array.
        """
        logger.info(f"Saving results to {output_dir}/")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        self._save_impact_scores(output_path, impact_scores)
        self._save_classification_map(output_path, classification_map)
        self._save_residual_map(output_path, residual_map)
        
        logger.info("All results saved successfully")
    
    def _save_impact_scores(
        self,
        output_path: Path,
        impact_scores: pd.DataFrame
    ) -> None:
        """
        Save Impact Score CSV files.
        
        Args:
            output_path: Output directory path.
            impact_scores: DataFrame with Impact Scores.
        """
        if impact_scores is None or impact_scores.empty:
            logger.warning("No impact scores to save")
            return
        
        detailed_path = output_path / "impact_scores_detailed.csv"
        impact_scores.to_csv(detailed_path, index=False, float_format='%.6f')
        logger.info(f"Saved detailed scores: {detailed_path}")
        
        simple_cols = ['Anomaly_ID', 'Type', 'Centroid_Row', 'Centroid_Col', 'IS']
        simple_scores = impact_scores[simple_cols]
        simple_path = output_path / "impact_scores.csv"
        simple_scores.to_csv(simple_path, index=False, float_format='%.6f')
        logger.info(f"Saved simplified scores: {simple_path}")
    
    def _save_classification_map(
        self,
        output_path: Path,
        classification_map: np.ndarray
    ) -> None:
        """
        Saves the classification map as a GeoTIFF file.
        
        Args:
            output_path: Output directory path.
            classification_map: Classification map array.
        """
        profile = self.raster_meta['profile'].copy()
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })
        
        output_file = output_path / "anomaly_classification.tif"
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(classification_map, 1)
        
        logger.info(f"Saved classification map: {output_file}")
    
    def _save_residual_map(
        self,
        output_path: Path,
        residual_map: np.ndarray
    ) -> None:
        """
        Saves the LST residual map as a GeoTIFF file.
        
        Args:
            output_path: Output directory path.
            residual_map: LST residual array.
        """
        profile = self.raster_meta['profile'].copy()
        profile.update({
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw'
        })
        
        output_file = output_path / "lst_residuals.tif"
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(residual_map.astype(np.float32), 1)
        
        logger.info(f"Saved residual map: {output_file}")
