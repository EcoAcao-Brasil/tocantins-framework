"""
Landsat Imagery Preprocessing Module with User-Defined Band Mapping

Supports Landsat 5/7/8/9 Level-2 Collection 2 imagery through flexible band mapping.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

logger = logging.getLogger(__name__)


class LandsatPreprocessor:
    """
    Preprocessor for Landsat imagery with user-defined band mapping.
    
    Parameters
    ----------
    band_mapping : dict, optional
        Dictionary mapping common names to band descriptions.
        If None, uses default Landsat 8/9 mapping.
        
        Required keys: 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 
                       'thermal', 'qa_pixel'
    """
    
    DEFAULT_BAND_MAPPING = {
        'coastal': 'SR_B1',
        'blue': 'SR_B2',
        'green': 'SR_B3',
        'red': 'SR_B4',
        'nir': 'SR_B5',
        'swir1': 'SR_B6',
        'swir2': 'SR_B7',
        'qa_aerosol': 'SR_QA_AEROSOL',
        'thermal': 'ST_B10',
        'qa_pixel': 'QA_PIXEL'
    }
    
    LST_SCALE_FACTOR = 0.00341802
    LST_OFFSET = 149.0
    KELVIN_TO_CELSIUS = 273.15
    
    def __init__(self, band_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize preprocessor with band mapping.
        
        Parameters
        ----------
        band_mapping : dict, optional
            User-defined band mapping. Uses Landsat 8/9 default if None.
        """
        self.band_mapping = band_mapping or self.DEFAULT_BAND_MAPPING.copy()
        
        required = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal']
        missing = [b for b in required if b not in self.band_mapping]
        
        if missing:
            raise ValueError(
                f"Missing required bands in mapping: {missing}. "
                f"Required: {required}"
            )
        
        self.raster_meta = {}
        self._lst_2d = None
        self._valid_mask_2d = None
        
        logger.info("LandsatPreprocessor initialized")
        for key in required:
            logger.debug(f"  {key}: {self.band_mapping[key]}")
    
    def load_imagery(self, tif_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and preprocess Landsat imagery from GeoTIFF file.
        
        Parameters
        ----------
        tif_path : str
            Path to Landsat GeoTIFF file.
        
        Returns
        -------
        data : pd.DataFrame
            DataFrame with columns: x, y, row, col, LST, NDVI, NDWI, NDBI, NDBSI
        metadata : dict
            Geospatial metadata dictionary.
        """
        logger.info(f"Loading Landsat imagery from: {tif_path}")
        
        with rasterio.open(tif_path) as src:
            bands = src.read().astype(np.float64)
            
            logger.debug(f"Detected {bands.shape[0]} bands in GeoTIFF")
            
            self.raster_meta = {
                'transform': src.transform,
                'profile': src.profile,
                'height': src.height,
                'width': src.width,
                'pixel_size': abs(src.transform[0]),
                'crs': src.crs,
                'bounds': src.bounds
            }
            
            band_arrays = self._read_bands_by_name(src, bands)
            
            lst = self._convert_to_lst(
                band_arrays['thermal'],
                band_arrays.get('qa_pixel', band_arrays['blue'])
            )
            self._lst_2d = lst
            
            indices = self._calculate_spectral_indices(
                band_arrays['blue'],
                band_arrays['green'],
                band_arrays['red'],
                band_arrays['nir'],
                band_arrays['swir1'],
                band_arrays['swir2']
            )
            
            data = self._create_dataframe(lst, indices)
            
            valid_mask = data.notna().all(axis=1)
            data = data[valid_mask].reset_index(drop=True)
            self._valid_mask_2d = np.isfinite(lst)
            
            n_valid = len(data)
            n_total = self.raster_meta['height'] * self.raster_meta['width']
            pct_valid = 100 * n_valid / n_total
            
            logger.info(f"Loaded {n_valid:,} valid pixels ({pct_valid:.1f}% of image)")
        
        return data, self.raster_meta
    
    def _read_bands_by_name(
        self,
        src: rasterio.DatasetReader,
        bands: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Read bands using user-defined mapping and band descriptions."""
        descriptions = list(src.descriptions or [])
        
        if not descriptions:
            raise ValueError("Raster has no band descriptions")
        
        logger.debug(f"Available bands: {descriptions}")
        
        band_arrays = {}
        
        for common_name, band_name in self.band_mapping.items():
            if band_name not in descriptions:
                if common_name in ['qa_aerosol', 'coastal']:
                    logger.debug(f"Optional band {band_name} not found, skipping")
                    continue
                
                raise ValueError(
                    f"Required band '{band_name}' (for {common_name}) not found. "
                    f"Available: {descriptions}"
                )
            
            band_idx = descriptions.index(band_name)
            band_arrays[common_name] = bands[band_idx]
            
            logger.debug(f"  {common_name}: {band_name} at index {band_idx}")
        
        thermal_data = band_arrays['thermal']
        thermal_min = np.nanmin(thermal_data)
        thermal_max = np.nanmax(thermal_data)
        
        logger.debug(f"Thermal band ({self.band_mapping['thermal']}) range: "
                    f"[{thermal_min:.2f}, {thermal_max:.2f}]")
        
        if thermal_max < 100:
            logger.warning(
                f"Thermal band maximum ({thermal_max:.2f}) is unexpectedly low. "
                f"Expected ~250-350 (Kelvin) or ~10000-15000 (DN). "
                f"Verify correct band is mapped."
            )
        
        return band_arrays
    
    def _convert_to_lst(
        self,
        st_dn: np.ndarray,
        qa_band: np.ndarray
    ) -> np.ndarray:
        """Convert thermal band to Land Surface Temperature."""
        logger.debug("Converting thermal band to LST")
        
        st_kelvin = st_dn * self.LST_SCALE_FACTOR + self.LST_OFFSET
        lst = st_kelvin - self.KELVIN_TO_CELSIUS
        
        lst[np.isin(qa_band, [0, 1]) | np.isnan(st_dn)] = np.nan
        
        valid_lst = lst[np.isfinite(lst)]
        if valid_lst.size > 0:
            logger.debug(f"LST range: {valid_lst.min():.2f}°C to {valid_lst.max():.2f}°C")
            logger.debug(f"LST mean: {valid_lst.mean():.2f}°C (std={valid_lst.std():.2f}°C)")
        
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
        """Calculate spectral indices from surface reflectance bands."""
        logger.debug("Calculating spectral indices")
        
        with np.errstate(divide='ignore', invalid='ignore'):
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
        """Create structured DataFrame with spatial coordinates."""
        logger.debug("Creating structured DataFrame")
        
        height = self.raster_meta['height']
        width = self.raster_meta['width']
        
        rows, cols = np.mgrid[0:height, 0:width]
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
        if self._lst_2d is None:
            raise RuntimeError("No LST data available. Call load_imagery() first.")
        return self._lst_2d
    
    def get_valid_mask_2d(self) -> np.ndarray:
        """Get 2D valid pixel mask."""
        if self._valid_mask_2d is None:
            raise RuntimeError("No mask available. Call load_imagery() first.")
        return self._valid_mask_2d
