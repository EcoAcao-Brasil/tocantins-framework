# Band Mapping Guide

This document provides detailed information about band mapping for different Landsat sensor versions in the Tocantins Framework.

## Overview

The Tocantins Framework supports multiple Landsat sensor versions through flexible band mapping. Band mapping allows you to specify which bands in your Landsat imagery correspond to the required spectral channels.

## Required Bands

All configurations must include these bands:

| Common Name | Purpose | Typical Wavelength |
|-------------|---------|-------------------|
| `blue` | Blue spectral band | 0.45-0.52 μm |
| `green` | Green spectral band | 0.52-0.60 μm |
| `red` | Red spectral band | 0.63-0.69 μm |
| `nir` | Near-Infrared | 0.77-0.90 μm |
| `swir1` | Shortwave Infrared 1 | 1.55-1.75 μm |
| `swir2` | Shortwave Infrared 2 | 2.08-2.35 μm |
| `thermal` | Thermal Infrared | 10.4-12.5 μm |
| `qa_pixel` | Quality Assessment | N/A |

## Landsat-Specific Mappings

### Landsat 8 and 9 (OLI/TIRS)

**Default configuration** - No mapping required!

```python
# Automatic for Landsat 8/9
from tocantins_framework import calculate_tocantins_framework

calculator = calculate_tocantins_framework(
    tif_path="LC08_scene.tif",
    output_dir="results"
)
```

Internal mapping (for reference):
```python
{
    'blue': 'SR_B2',
    'green': 'SR_B3',
    'red': 'SR_B4',
    'nir': 'SR_B5',
    'swir1': 'SR_B6',
    'swir2': 'SR_B7',
    'thermal': 'ST_B10',  # Band 10
    'qa_pixel': 'QA_PIXEL'
}
```

### Landsat 7 (ETM+)

```python
l7_mapping = {
    'blue': 'SR_B1',
    'green': 'SR_B2',
    'red': 'SR_B3',
    'nir': 'SR_B4',
    'swir1': 'SR_B5',
    'swir2': 'SR_B7',
    'thermal': 'ST_B6',  # Band 6
    'qa_pixel': 'QA_PIXEL'
}

calculator = calculate_tocantins_framework(
    tif_path="LE07_scene.tif",
    band_mapping=l7_mapping,
    output_dir="results"
)
```

### Landsat 5 (TM)

```python
l5_mapping = {
    'blue': 'SR_B1',
    'green': 'SR_B2',
    'red': 'SR_B3',
    'nir': 'SR_B4',
    'swir1': 'SR_B5',
    'swir2': 'SR_B7',
    'thermal': 'ST_B6',  # Band 6
    'qa_pixel': 'QA_PIXEL'
}

calculator = calculate_tocantins_framework(
    tif_path="LT05_scene.tif",
    band_mapping=l5_mapping,
    output_dir="results"
)
```

### Landsat 4 (TM)

Same as Landsat 5:

```python
l4_mapping = {
    'blue': 'SR_B1',
    'green': 'SR_B2',
    'red': 'SR_B3',
    'nir': 'SR_B4',
    'swir1': 'SR_B5',
    'swir2': 'SR_B7',
    'thermal': 'ST_B6',
    'qa_pixel': 'QA_PIXEL'
}
```

## Band Name Resolution

The framework identifies bands by their **description** (metadata), not by position. This means:

1. Band descriptions must match exactly (case-sensitive)
2. Band order in the file doesn't matter
3. Extra bands are ignored

### Finding Band Descriptions

To check band descriptions in your GeoTIFF:

```python
import rasterio

with rasterio.open("your_scene.tif") as src:
    descriptions = src.descriptions
    for i, desc in enumerate(descriptions, 1):
        print(f"Band {i}: {desc}")
```

## Spectral Index Calculations

The framework calculates these indices from the mapped bands:

### NDVI (Normalized Difference Vegetation Index)
```
NDVI = (NIR - Red) / (NIR + Red)
```
Range: [-1, 1]  
Vegetation typically > 0.2

### NDWI (Normalized Difference Water Index)
```
NDWI = (Green - NIR) / (Green + NIR)
```
Range: [-1, 1]  
Water typically > 0

### NDBI (Normalized Difference Built-up Index)
```
NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
```
Range: [-1, 1]  
Built-up areas typically > 0

### NDBSI (Normalized Difference Bareness and Soil Index)
```
NDBSI = ((Red + SWIR1) - (NIR + Blue)) / ((Red + SWIR1) + (NIR + Blue))
```
Range: [-1, 1]  
Bare soil typically > 0

## Thermal Band Considerations

### Critical Differences

**Landsat 8/9**: Uses Band 10 (ST_B10)
- Wavelength: 10.6-11.19 μm
- Better thermal resolution

**Landsat 4/5/7**: Uses Band 6 (ST_B6)
- Wavelength: 10.4-12.5 μm (L5/7) or 10.4-11.5 μm (L4)
- Single thermal band

### Temperature Conversion

The framework applies USGS-standard conversion:
```python
LST_Kelvin = DN × 0.00341802 + 149.0
LST_Celsius = LST_Kelvin - 273.15
```

This conversion is consistent across all Landsat versions for Level-2 Collection 2 data.

## Validation and Error Checking

The framework performs automatic validation:

### Required Band Check
```python
# Missing bands trigger error
ValueError: Missing required bands in mapping: ['thermal']
```

### Band Description Check
```python
# Incorrect band name triggers error
ValueError: Required band 'ST_B11' (for thermal) not found.
Available: ['SR_B1', 'SR_B2', ...]
```

### Thermal Band Range Warning
```python
# Unusual thermal values trigger warning
WARNING: Thermal band maximum (45.23) is unexpectedly low.
Expected ~250-350 (Kelvin) or ~10000-15000 (DN).
Verify correct band is mapped.
```

## Advanced Usage

### Custom Pre-Processing

If your data has non-standard band names:

```python
custom_mapping = {
    'blue': 'Band_2_Blue',
    'green': 'Band_3_Green',
    'red': 'Band_4_Red',
    'nir': 'Band_5_NIR',
    'swir1': 'Band_6_SWIR1',
    'swir2': 'Band_7_SWIR2',
    'thermal': 'Band_10_Thermal',
    'qa_pixel': 'QA_Pixel'
}
```

### Optional Bands

Some bands are optional and won't cause errors if missing:
- `coastal` (SR_B1 in L8/9)
- `qa_aerosol` (SR_QA_AEROSOL)

## Troubleshooting

### Issue: "No band descriptions"
**Cause**: GeoTIFF lacks metadata  
**Solution**: Ensure file is official USGS Level-2 product

### Issue: "Thermal band range unexpected"
**Cause**: Wrong band mapped as thermal  
**Solution**: Verify thermal band name (ST_B10 vs ST_B6)

### Issue: "Missing required bands"
**Cause**: Incomplete band mapping  
**Solution**: Include all required bands in mapping dict

## References

- [Landsat 8-9 Data Users Handbook](https://www.usgs.gov/landsat-missions/landsat-8-data-users-handbook)
- [Landsat 4-7 Data Users Handbook](https://www.usgs.gov/landsat-missions/landsat-7-data-users-handbook)
- [Landsat Collection 2 Surface Temperature](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature)

## Quick Reference Table

| Landsat | Blue | Green | Red | NIR | SWIR1 | SWIR2 | Thermal |
|---------|------|-------|-----|-----|-------|-------|---------|
| L4/5 TM | B1 | B2 | B3 | B4 | B5 | B7 | B6 |
| L7 ETM+ | B1 | B2 | B3 | B4 | B5 | B7 | B6 |
| L8/9 OLI/TIRS | B2 | B3 | B4 | B5 | B6 | B7 | B10 |
