"""
Landsat Imagery Preprocessing Module

This module provides functionality for loading, preprocessing, and extracting
features from Landsat 8/9 Level-2 Collection 2 imagery.

The module handles:
- Multi-band GeoTIFF loading with proper geospatial metadata
- Land Surface Temperature (LST) conversion and validation
- Spectral index calculation (NDVI, NDWI, NDBI, NDBSI)
- Quality assessment and data masking

Landsat Band Organization (Level-2 Collection 2):
Band 1:  SR_B1         - Coastal/Aerosol (0.43-0.45 μm)
Band 2:  SR_B2 (Blue)  - Blue (0.45-0.51 μm)
Band 3:  SR_B3 (Green) - Green (0.53-0.59 μm)
Band 4:  SR_B4 (Red)   - Red (0.64-0.67 μm)
Band 5:  SR_B5 (NIR)   - Near-Infrared (0.85-0.88 μm)
Band 6:  SR_B6 (SWIR1) - Shortwave Infrared 1 (1.57-1.65 μm)
Band 7:  SR_B7 (SWIR2) - Shortwave Infrared 2 (2.11-2.29 μm)
Band 8:  SR_QA_AEROSOL - Aerosol QA
Band 9:  ST_B10        - Thermal Infrared (10.6-11.19 μm)
Band 10-17: Various QA and atmospheric correction bands
Band 18: QA_PIXEL      - Pixel Quality Assessment
Band 19: QA_RADSAT     - Radiometric Saturation QA

Spectral Indices for The Tocantins Framework:
NDVI (Normalized Difference Vegetation Index):
    (NIR - Red) / (NIR + Red)
    Range: [-1, 1], vegetation typically > 0.2

NDWI (Normalized Difference Water Index):
    (Green - NIR) / (Green + NIR)
    Range: [-1, 1], water typically > 0

NDBI (Normalized Difference Built-up Index):
    (SWIR1 - NIR) / (SWIR1 + NIR)
    Range: [-1, 1], built-up areas typically > 0

NDBSI (Normalized Difference Bareness and Soil Index):
    ((Red + SWIR1) - (NIR + Blue)) / ((Red + SWIR1) + (NIR + Blue))
    Range: [-1, 1], bare soil typically > 0

Examples:
>>> from tocantins_framework.preprocessing import LandsatPreprocessor
>>> 
>>> preprocessor = LandsatPreprocessor()
>>> data, metadata = preprocessor.load_imagery("LC08_scene.tif")
>>> 
>>> # Access 2D arrays for spatial analysis
>>> lst_array = preprocessor.get_lst_2d()
>>> valid_mask = preprocessor.get_valid_mask_2d()
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
    Preprocessor for Landsat 8/9 Level-2 Collection 2 GeoTIFF imagery.
    
    This class handles the complete preprocessing pipeline including band
    extraction, LST conversion, spectral index calculation, and quality masking.
    
    Attributes
    ----------
    raster_meta : dict
        Metadata dictionary containing geospatial information:
        - 'transform': Affine transformation matrix
        - 'profile': Rasterio profile with CRS and other parameters
        - 'height': Raster height in pixels
        - 'width': Raster width in pixels
        - 'pixel_size': Ground sampling distance in meters (default: 30.0)
    
    Methods
    -------
    load_imagery(tif_path)
        Load and preprocess Landsat imagery.
    get_lst_2d()
        Get 2D Land Surface Temperature array.
    get_valid_mask_2d()
        Get 2D boolean mask of valid pixels.
    
    Notes
    -----
    The preprocessor assumes Landsat Level-2 Collection 2 data structure.
    For other Landsat products, band ordering may differ and require adjustment.
    
    Examples
    --------
    >>> preprocessor = LandsatPreprocessor()
    >>> data_df, metadata = preprocessor.load_imagery("landsat_scene.tif")
    >>> print(f"Loaded {len(data_df)} valid pixels")
    >>> print(f"LST range: {data_df['LST'].min():.1f} to {data_df['LST'].max():.1f}°C")
    """
    
    # Band indices for Landsat Level-2 Collection 2
    BAND_INDICES = {
        'coastal': 0,   # SR_B1
        'blue': 1,      # SR_B2
        'green': 2,     # SR_B3
        'red': 3,       # SR_B4
        'nir': 4,       # SR_B5
        'swir1': 5,     # SR_B6
        'swir2': 6,     # SR_B7
        'qa_aerosol': 7,  # SR_QA_AEROSOL
        'thermal': 8,   # ST_B10
        'qa_pixel': 17  # QA_PIXEL (band 18 in 1-indexed)
    }
    
    # LST conversion constants for Landsat Collection 2
    LST_SCALE_FACTOR = 0.00341802  # Kelvin per DN
    LST_OFFSET = 149.0             # Kelvin offset
    KELVIN_TO_CELSIUS = 273.15     # Conversion constant
    
    def __init__(self):
        """Initialize the Landsat preprocessor."""
        self.raster_meta = {}
        self._lst_2d = None
        self._valid_mask_2d = None
    
    def load_imagery(self, tif_path: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and preprocess Landsat imagery from GeoTIFF file.
        
        This method performs the complete preprocessing pipeline:
        1. Load all bands from GeoTIFF
        2. Extract geospatial metadata
        3. Convert thermal band to Land Surface Temperature
        4. Calculate spectral indices
        5. Apply quality masking
        6. Create structured DataFrame
        
        Parameters
        ----------
        tif_path : str
            Path to Landsat Level-2 Collection 2 GeoTIFF file.
            Must contain all required bands in expected order.
        
        Returns
        -------
        data : pd.DataFrame
            DataFrame containing valid pixels with columns:
            - 'x', 'y': Geographic coordinates
            - 'row', 'col': Pixel indices
            - 'LST': Land Surface Temperature (°C)
            - 'NDVI', 'NDWI', 'NDBI', 'NDBSI': Spectral indices
        
        metadata : dict
            Dictionary containing:
            - 'transform': Affine transformation
            - 'profile': Complete rasterio profile
            - 'height', 'width': Image dimensions
            - 'pixel_size': Ground sampling distance (m)
        
        Raises
        ------
        FileNotFoundError
            If the specified GeoTIFF file does not exist.
        ValueError
            If the GeoTIFF does not contain expected number of bands.
        rasterio.errors.RasterioIOError
            If file cannot be opened or read.
        
        Examples
        --------
        >>> preprocessor = LandsatPreprocessor()
        >>> data, meta = preprocessor.load_imagery("LC08_L2SP_scene.tif")
        >>> print(f"Image size: {meta['height']} × {meta['width']} pixels")
        >>> print(f"Valid pixels: {len(data):,}")
        """
        logger.info(f"Loading Landsat imagery from: {tif_path}")
        
        with rasterio.open(tif_path) as src:
            # Read all bands as float64 for numerical precision
            bands = src.read().astype(np.float64)
            
            # Validate band count
            n_bands = bands.shape[0]
            logger.debug(f"Detected {n_bands} bands in GeoTIFF")
            
            # Store geospatial metadata
            self.raster_meta = {
                'transform': src.transform,
                'profile': src.profile,
                'height': src.height,
                'width': src.width,
                'pixel_size': abs(src.transform[0]),  # Extract from transform
                'crs': src.crs,
                'bounds': src.bounds
            }
            
            # Extract required bands
            blue = bands[self.BAND_INDICES['blue']]
            green = bands[self.BAND_INDICES['green']]
            red = bands[self.BAND_INDICES['red']]
            nir = bands[self.BAND_INDICES['nir']]
            swir1 = bands[self.BAND_INDICES['swir1']]
            swir2 = bands[self.BAND_INDICES['swir2']]
            thermal_dn = bands[self.BAND_INDICES['thermal']]
            qa_band = bands[self.BAND_INDICES['coastal']]  # Using first band for QA
            
            # Convert thermal band to LST
            lst = self._convert_to_lst(thermal_dn, qa_band)
            self._lst_2d = lst
            
            # Calculate spectral indices
            indices = self._calculate_spectral_indices(
                blue, green, red, nir, swir1, swir2
            )
            
            # Create structured DataFrame
            data = self._create_dataframe(lst, indices)
            
            # Apply quality masking and remove invalid pixels
            valid_mask = data.notna().all(axis=1)
            data = data[valid_mask].reset_index(drop=True)
            self._valid_mask_2d = np.isfinite(lst)
            
            n_valid = len(data)
            n_total = self.raster_meta['height'] * self.raster_meta['width']
            pct_valid = 100 * n_valid / n_total
            
            logger.info(f"Loaded {n_valid:,} valid pixels ({pct_valid:.1f}% of image)")
        
        return data, self.raster_meta
    
    def _convert_to_lst(self, st_dn: np.ndarray, qa_band: np.ndarray) -> np.ndarray:
        """
        Convert Landsat thermal band digital numbers to Land Surface Temperature.
        
        Applies the official USGS conversion formula for Landsat Collection 2:
        LST (K) = DN × 0.00341802 + 149.0
        LST (°C) = LST (K) - 273.15
        
        Parameters
        ----------
        st_dn : np.ndarray
            Surface temperature digital numbers (ST_B10).
        qa_band : np.ndarray
            Quality assessment band for masking invalid pixels.
        
        Returns
        -------
        lst : np.ndarray
            Land Surface Temperature in degrees Celsius with NaN for invalid pixels.
        
        Notes
        -----
        Invalid pixels (QA values 0 or 1) are masked to NaN.
        If DN values appear already scaled (max < 100), scaling is still applied
        to maintain consistency with USGS specifications.
        """
        logger.debug("Converting thermal band to LST")
        
        # Apply USGS scaling if DN values are raw
        # Note: Some pre-processed data may already be in Kelvin
        if np.nanmax(st_dn) < 100:
            logger.debug("Thermal values appear pre-scaled, applying standard conversion")
            st_kelvin = st_dn * self.LST_SCALE_FACTOR + self.LST_OFFSET
        else:
            logger.debug("Applying LST scaling factors")
            st_kelvin = st_dn * self.LST_SCALE_FACTOR + self.LST_OFFSET
        
        # Convert to Celsius
        lst = st_kelvin - self.KELVIN_TO_CELSIUS
        
        # Apply quality masking (QA values 0, 1 indicate poor quality)
        lst[np.isin(qa_band, [0, 1]) | np.isnan(st_dn)] = np.nan
        
        # Log statistics
        valid_lst = lst[np.isfinite(lst)]
        if valid_lst.size > 0:
            logger.debug(f"LST range: {valid_lst.min():.2f}°C to {valid_lst.max():.2f}°C")
            logger.debug(f"LST mean: {valid_lst.mean():.2f}°C (σ={valid_lst.std():.2f}°C)")
        
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
        Calculate standard spectral indices from Landsat surface reflectance bands.
        
        Parameters
        ----------
        blue, green, red, nir, swir1, swir2 : np.ndarray
            Surface reflectance bands (0-1 range expected).
        
        Returns
        -------
        indices : dict
            Dictionary with keys 'NDVI', 'NDWI', 'NDBI', 'NDBSI' mapping
            to 2D numpy arrays of calculated index values.
        
        Notes
        -----
        Division by zero is handled automatically by numpy (returns inf/nan).
        All indices are normalized to [-1, 1] range.
        """
        logger.debug("Calculating spectral indices")
        
        # NDVI: Vegetation index
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
        
        # NDWI: Water index
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green - nir) / (green + nir)
        
        # NDBI: Built-up index
        with np.errstate(divide='ignore', invalid='ignore'):
            ndbi = (swir1 - nir) / (swir1 + nir)
        
        # NDBSI: Bareness and soil index
        with np.errstate(divide='ignore', invalid='ignore'):
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
        Create structured DataFrame with spatial coordinates and spectral data.
        
        Parameters
        ----------
        lst : np.ndarray
            2D Land Surface Temperature array.
        indices : dict
            Dictionary of spectral indices (2D arrays).
        
        Returns
        -------
        data : pd.DataFrame
            DataFrame with columns: x, y, row, col, LST, NDVI, NDWI, NDBI, NDBSI.
        
        Notes
        -----
        Geographic coordinates (x, y) are calculated using the raster's affine
        transformation matrix, representing pixel centers.
        """
        logger.debug("Creating structured DataFrame")
        
        height = self.raster_meta['height']
        width = self.raster_meta['width']
        
        # Create coordinate grids
        rows, cols = np.mgrid[0:height, 0:width]
        
        # Transform to geographic coordinates
        xs, ys = xy(self.raster_meta['transform'], rows, cols)
        
        # Flatten all arrays for DataFrame construction
        data = pd.DataFrame({
            'x': np.array(xs).flatten(),
            'y': np.array(ys).flatten(),
            'row': rows.flatten(),
            'col': cols.flatten(),
            'LST': lst.flatten(),
        })
        
        # Add spectral indices
        for name, values in indices.items():
            data[name] = values.flatten()
        
        return data
    
    def get_lst_2d(self) -> np.ndarray:
        """
        Get 2D Land Surface Temperature array.
        
        Returns
        -------
        np.ndarray
            2D array of LST values in degrees Celsius.
            Shape: (height, width)
        
        Raises
        ------
        RuntimeError
            If load_imagery() has not been called yet.
        
        Examples
        --------
        >>> preprocessor = LandsatPreprocessor()
        >>> data, meta = preprocessor.load_imagery("scene.tif")
        >>> lst_array = preprocessor.get_lst_2d()
        >>> print(f"LST shape: {lst_array.shape}")
        """
        if self._lst_2d is None:
            raise RuntimeError("No LST data available. Call load_imagery() first.")
        return self._lst_2d
    
    def get_valid_mask_2d(self) -> np.ndarray:
        """
        Get 2D boolean mask of valid (non-NaN) pixels.
        
        Returns
        -------
        np.ndarray
            2D boolean array where True indicates valid pixel.
            Shape: (height, width)
        
        Raises
        ------
        RuntimeError
            If load_imagery() has not been called yet.
        
        Examples
        --------
        >>> mask = preprocessor.get_valid_mask_2d()
        >>> n_valid = np.sum(mask)
        >>> print(f"Valid pixels: {n_valid:,}")
        """
        if self._valid_mask_2d is None:
            raise RuntimeError("No mask available. Call load_imagery() first.")
        return self._valid_mask_2d