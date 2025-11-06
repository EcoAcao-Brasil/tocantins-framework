"""
Preprocessing module for Landsat imagery.

Handles loading, band extraction, and spectral index calculation, considering the following bands:
Band 1: SR_B1, Band 2: SR_B2, Band 3: SR_B3, Band 4: SR_B4, Band 5: SR_B5, Band 6: SR_B6,
Band 7: SR_B7, Band 8: SR_QA_AEROSOL, Band 9: ST_B10, Band 10: ST_ATRAN, Band 11: ST_CDIST,
Band 12: ST_DRAD, Band 13: ST_EMIS, Band 14: ST_EMSD, Band 15: ST_QA, Band 16: ST_TRAD
Band 17: ST_URAD, Band 18: QA_PIXEL, Band 19: QA_RADSAT, Band 20 (manually forced into data acquisition): timestamp
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

logger = logging.getLogger(__name__)


class LandsatPreprocessor:
    """
    Preprocessor for Landsat 8 GeoTIFF imagery.
    
    Handles band extraction, LST conversion, and spectral index computation.
    """
    
    def __init__(self):
        self.raster_meta = {}
        self._lst_2d = None
        self._valid_mask_2d = None
    
    def load_imagery(self, tif_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and preprocess Landsat imagery.
        
        Args:
            tif_path: Path to Landsat GeoTIFF file.
            
        Returns:
            Tuple of (DataFrame with pixel data, metadata dictionary).
        """
        logger.info("Loading Landsat imagery")
        
        with rasterio.open(tif_path) as src:
            bands = src.read().astype(np.float64)
            
            self.raster_meta = {
                'transform': src.transform,
                'profile': src.profile,
                'height': src.height,
                'width': src.width,
                'pixel_size': 30.0
            }
            
            blue, green, red, nir, swir1, swir2, st_dn = (
                bands[1], bands[2], bands[3], bands[4], 
                bands[5], bands[6], bands[8]
            )
            
            lst = self._convert_to_lst(st_dn, bands[0])
            self._lst_2d = lst
            
            indices = self._calculate_spectral_indices(
                blue, green, red, nir, swir1, swir2
            )
            
            data = self._create_dataframe(lst, indices)
            
            valid_mask = data.notna().all(axis=1)
            data = data[valid_mask].reset_index(drop=True)
            self._valid_mask_2d = np.isfinite(lst)
            
            logger.info(f"Loaded {len(data):,} valid pixels")
        
        return data, self.raster_meta
    
    def _convert_to_lst(self, st_dn: np.ndarray, qa_band: np.ndarray) -> np.ndarray:
        """
        Convert thermal band to Land Surface Temperature in Celsius.
        
        Args:
            st_dn: Surface temperature digital number.
            qa_band: Quality assessment band.
            
        Returns:
            LST in degrees Celsius.
        """
        if np.nanmax(st_dn) < 100:
            st_dn = st_dn * 0.00341802 + 149.0
        
        lst = st_dn - 273.15
        lst[np.isin(qa_band, [0, 1]) | np.isnan(st_dn)] = np.nan
        
        return lst
    
    def _calculate_spectral_indices(
        self,
        blue: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir1: np.ndarray,
        swir2: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calculate spectral indices from Landsat bands.
        
        Args:
            blue, green, red, nir, swir1, swir2: Landsat band arrays.
            
        Returns:
            Dictionary of spectral indices.
        """
        ndvi = (nir - red) / (nir + red)
        ndwi = (green - nir) / (green + nir)
        ndbi = (swir1 - nir) / (swir1 + nir)
        ndbsi = ((red + swir1) - (nir + blue)) / ((red + swir1) + (nir + blue))
        
        return {
            'NDVI': ndvi,
            'NDWI': ndwi,
            'NDBI': ndbi,
            'NDBSI': ndbsi
        }
    
    def _create_dataframe(
        self,
        lst: np.ndarray,
        indices: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Create DataFrame with spatial coordinates and spectral data.
        
        Args:
            lst: Land Surface Temperature array.
            indices: Dictionary of spectral indices.
            
        Returns:
            DataFrame with all pixel data.
        """
        rows, cols = np.mgrid[0:self.raster_meta['height'], 0:self.raster_meta['width']]
        xs, ys = xy(self.raster_meta['transform'], rows, cols)
        
        data = pd.DataFrame({
            'x': np.array(xs).flatten(),
            'y': np.array(ys).flatten(),
            'row': rows.flatten(),
            'col': cols.flatten(),
            'LST': lst.flatten(),
        })
        
        for name, values in indices.items():
            data[name] = values.flatten()
        
        return data
    
    def get_lst_2d(self) -> np.ndarray:
        """Get 2D LST array."""
        return self._lst_2d
    
    def get_valid_mask_2d(self) -> np.ndarray:
        """Get 2D valid pixel mask."""
        return self._valid_mask_2d
