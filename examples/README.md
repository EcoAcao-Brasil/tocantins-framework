# Examples

This directory contains practical examples demonstrating various usage scenarios for the Tocantins Framework.

## Available Examples

### 1. Basic Landsat 8/9 Analysis
**File:** `basic_landsat8_analysis.py`

The simplest way to get started. Analyzes Landsat 8/9 imagery using default settings.

```bash
python examples/basic_landsat8_analysis.py
```

**What it demonstrates:**
- One-line analysis with `calculate_tocantins_framework()`
- Default Landsat 8/9 band mapping (automatic)
- Accessing results (feature set, impact scores, severity scores)
- Output file structure

### 2. Landsat 5 Custom Mapping
**File:** `landsat5_custom_mapping.py`

Shows how to analyze Landsat 5 imagery with custom band mapping.

```bash
python examples/landsat5_custom_mapping.py
```

**What it demonstrates:**
- Custom band mapping for different Landsat versions
- Landsat 5-specific band configuration
- Working with different thermal bands (ST_B6 vs ST_B10)
- Analyzing results by anomaly type (hot vs cold)

### 3. Advanced Configuration
**File:** `advanced_configuration.py`

Full control over all framework parameters for specialized scenarios.

```bash
python examples/advanced_configuration.py
```

**What it demonstrates:**
- Custom spatial morphology parameters
- Random Forest hyperparameter tuning
- Metrics calculation parameters
- Adjusting detection sensitivity (k_threshold)
- Detailed logging configuration
- Statistical analysis of results

## Data Requirements

These examples expect Landsat Level-2 Collection 2 GeoTIFF files in a `data/` directory:

```
examples/
├── data/
│   ├── LC08_L2SP_example.tif  # Landsat 8/9 scene
│   └── LT05_L2SP_example.tif  # Landsat 5 scene (optional)
├── basic_landsat8_analysis.py
├── landsat5_custom_mapping.py
└── advanced_configuration.py
```

## Downloading Landsat Data

You can obtain Landsat imagery from:

1. **USGS Earth Explorer**: https://earthexplorer.usgs.gov/
   - Select "Landsat" → "Landsat Collection 2 Level-2"
   - Choose your area of interest and date range
   - Download as GeoTIFF format

2. **Google Earth Engine**: https://earthengine.google.com/
   - Use the Earth Engine Code Editor
   - Export Landsat scenes to GeoTIFF

## Band Mapping Reference

### Landsat 8/9 (Default)
```python
# No mapping needed - works automatically
calculator = calculate_tocantins_framework(tif_path="LC08_scene.tif")
```

### Landsat 5/7
```python
band_mapping = {
    'blue': 'SR_B1',
    'green': 'SR_B2',
    'red': 'SR_B3',
    'nir': 'SR_B4',
    'swir1': 'SR_B5',
    'swir2': 'SR_B7',
    'thermal': 'ST_B6',  # Note: B6 for L5/L7, B10 for L8/L9
    'qa_pixel': 'QA_PIXEL'
}
```

## Output Files

All examples generate the following outputs:

### CSV Files
- `impact_scores.csv` - Simplified Impact Scores
- `impact_scores_detailed.csv` - Detailed Impact Score metrics
- `severity_scores.csv` - Core-level Severity Scores
- `ml_features.csv` - Complete feature set (IS + SS combined)

### GeoTIFF Files
- `anomaly_classification.tif` - Classification map
  - 0: Background
  - 1: Cold EAZ
  - 2: Hot EAZ
  - 3: Cold Core
  - 4: Hot Core
- `lst_residuals.tif` - LST residual map (°C)

## Customization Guide

### Adjusting Detection Sensitivity

**More Conservative** (fewer, more confident anomalies):
```python
k_threshold = 2.0  # Default is 1.5
```

**More Sensitive** (more anomalies, may include weaker signals):
```python
k_threshold = 1.0
```

### Spatial Processing

**Larger minimum anomaly size** (filter out small artifacts):
```python
spatial_params = {'min_anomaly_size': 10}  # Default is 1
```

**Aggressive agglutination** (merge nearby anomalies):
```python
spatial_params = {'agglutination_distance': 6}  # Default is 4
```

### Random Forest Tuning

**Higher accuracy** (slower):
```python
rf_params = {'n_estimators': 500, 'max_depth': 40}
```

**Faster processing** (lower accuracy):
```python
rf_params = {'n_estimators': 100, 'max_depth': 15}
```

## Support

For questions or issues:
- GitHub Issues: https://github.com/EcoAcao-Brasil/tocantins-framework/issues
- Email: isaque@ecoacaobrasil.org
