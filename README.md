# Tocantins Framework

A comprehensive Python framework for detecting and quantifying intra-urban thermal anomalies in Landsat imagery using machine learning and spatial analysis techniques.

## Overview

The Tocantins Framework provides researchers and practitioners with robust tools for analyzing urban heat islands and thermal anomalies using Landsat satellite imagery. The framework implements a scientifically rigorous methodology combining statistical analysis, machine learning, and spatial morphology to produce two complementary metrics:

- **Impact Score (IS)**: Quantifies the spatial extent and thermal significance of Extended Anomaly Zones (EAZs)
- **Severity Score (SS)**: Quantifies the thermal intensity of anomaly cores

These metrics enable data-driven urban heat intervention planning and climate adaptation strategies.

## Key Features

- **Multi-Landsat Support**: Works with Landsat 5, 7, 8, and 9 Level-2 Collection 2 imagery
- **Flexible Band Mapping**: User-defined or automatic band configuration
- **Automated Processing Pipeline**: End-to-end analysis from raw imagery to actionable metrics
- **Machine Learning Integration**: Random Forest-based residual analysis for robust anomaly detection
- **Spatial Coherence**: Morphological operations ensure spatially meaningful anomaly delineation
- **Standardized Metrics**: Reproducible, comparable thermal anomaly quantification
- **Configurable Parameters**: Customizable for different urban contexts
- **Scientific Rigor**: Implements peer-reviewed methodologies with full transparency

## Installation

### From PyPI (Recommended)

```bash
pip install tocantins-framework
```

### From Source

```bash
git clone https://github.com/EcoAcao-Brasil/tocantins-framework
cd Tocantins-Framework
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

### Landsat 8/9 (Default)

```python
from tocantins_framework import calculate_tocantins_framework

# Run complete analysis on Landsat 8/9 imagery
calculator = calculate_tocantins_framework(
    tif_path="path/to/LC08_scene.tif",
    output_dir="results"
)

# Access results
feature_set = calculator.get_feature_set()
print(feature_set[['Anomaly_ID', 'Type', 'IS', 'SS']])
```

### Landsat 5/7 (Custom Mapping)

```python
from tocantins_framework import calculate_tocantins_framework

# Define Landsat 5 band mapping
l5_mapping = {
    'blue': 'SR_B1',
    'green': 'SR_B2',
    'red': 'SR_B3',
    'nir': 'SR_B4',
    'swir1': 'SR_B5',
    'swir2': 'SR_B7',
    'thermal': 'ST_B6',
    'qa_pixel': 'QA_PIXEL'
}

# Run analysis with custom band mapping
calculator = calculate_tocantins_framework(
    tif_path="path/to/LT05_scene.tif",
    band_mapping=l5_mapping,
    output_dir="results"
)
```

### Advanced Configuration

```python
from tocantins_framework import TocantinsFrameworkCalculator

# Configure custom parameters
spatial_params = {
    'min_anomaly_size': 5,
    'agglutination_distance': 3,
    'connectivity': 2
}

rf_params = {
    'n_estimators': 300,
    'max_depth': 30,
    'random_state': 42
}

# Initialize with custom configuration
calculator = TocantinsFrameworkCalculator(
    k_threshold=1.5,
    spatial_params=spatial_params,
    rf_params=rf_params
)

# Run analysis
success = calculator.run_complete_analysis(
    tif_path="path/to/landsat_scene.tif",
    output_dir="results",
    save_results=True
)
```

## Input Data Requirements

### Supported Satellites
- **Landsat 8/9**: Works out-of-the-box (default configuration)
- **Landsat 5/7**: Requires custom band mapping

### Data Format
Landsat Level-2 Collection 2 GeoTIFF files containing:
- Surface Reflectance bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
- Surface Temperature band (Thermal)
- Quality Assessment band (QA_PIXEL)

### Data Sources
- [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- [Google Earth Engine](https://earthengine.google.com/)

## Band Mapping Reference

### Landsat 8/9 (Default)
No configuration needed - automatic detection.

### Landsat 5
```python
{
    'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3',
    'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7',
    'thermal': 'ST_B6', 'qa_pixel': 'QA_PIXEL'
}
```

### Landsat 7
```python
{
    'blue': 'SR_B1', 'green': 'SR_B2', 'red': 'SR_B3',
    'nir': 'SR_B4', 'swir1': 'SR_B5', 'swir2': 'SR_B7',
    'thermal': 'ST_B6', 'qa_pixel': 'QA_PIXEL'
}
```

## Output Files

### CSV Files
- `ml_features.csv`: Complete feature set with IS and SS metrics
- `impact_scores.csv`: Simplified Impact Scores
- `impact_scores_detailed.csv`: Detailed Impact Score breakdown
- `severity_scores.csv`: Core-level Severity Scores

### GeoTIFF Files
- `anomaly_classification.tif`: Spatial classification map
  - 0 = Background
  - 1 = Cold EAZ
  - 2 = Hot EAZ
  - 3 = Cold Core
  - 4 = Hot Core
- `lst_residuals.tif`: Land Surface Temperature residuals

## Examples

See the `examples/` directory for detailed usage examples:
- `basic_landsat8_analysis.py` - Simple Landsat 8/9 analysis
- `landsat5_custom_mapping.py` - Landsat 5 with custom bands
- `advanced_configuration.py` - Full parameter customization

## Methodology

### Processing Pipeline

1. **Preprocessing**
   - Band extraction and validation
   - LST conversion (Kelvin to Celsius)
   - Spectral index calculation (NDVI, NDWI, NDBI, NDBSI)

2. **Statistical Anomaly Detection**
   - Percentile-based threshold identification (2nd and 98th percentiles)
   - Initial hot/cold anomaly classification

3. **Machine Learning Refinement**
   - Random Forest regression on non-anomalous pixels
   - Residual calculation (observed - predicted LST)
   - Core anomaly refinement using residual thresholds

4. **Spatial Morphology**
   - Core unification through morphological operations
   - Extended Anomaly Zone (EAZ) growth
   - Spatial coherence enforcement

5. **Metrics Calculation**
   - Impact Score: IS = sign(ΔT) × log(1 + severity × area × continuity)
   - Severity Score: SS = sign(ΔT) × log(1 + thermal_intensity × area)

## Configuration Parameters

### Spatial Parameters

```python
spatial_params = {
    'min_anomaly_size': 1,          # Minimum pixels for valid anomaly
    'agglutination_distance': 4,    # Dilation radius for core merging
    'morphology_kernel_size': 3,    # Morphological operation kernel size
    'connectivity': 2                # Pixel connectivity (1=4-conn, 2=8-conn)
}
```

### Detection Parameters

```python
k_threshold = 1.5  # Residual threshold multiplier (k × σ_residual)
```

### Random Forest Parameters

```python
rf_params = {
    'n_estimators': 200,        # Number of trees
    'max_depth': 25,            # Maximum tree depth
    'min_samples_split': 8,     # Minimum samples to split node
    'min_samples_leaf': 4,      # Minimum samples per leaf
    'max_features': 'sqrt',     # Features per split
    'random_state': 42,         # Reproducibility seed
    'n_jobs': -1               # Parallel processing
}
```

## API Reference

### Main Classes

- `TocantinsFrameworkCalculator`: Main orchestrator for complete analysis
- `LandsatPreprocessor`: Imagery loading and preprocessing
- `AnomalyDetector`: Statistical and ML-based anomaly detection
- `MorphologyProcessor`: Spatial morphological operations
- `MetricsCalculator`: Impact and Severity Score calculation
- `ResultsWriter`: Output file generation

### Convenience Functions

- `calculate_tocantins_framework()`: One-line complete analysis

## Contributing

We welcome contributions from the scientific community! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tocantins_framework,
  author = {Borges, Isaque Carvalho},
  title = {Tocantins Framework: A Python Library for Assessment of Intra-Urban Thermal Anomaly},
  year = {2025},
  publisher = {EcoAção Brasil},
  url = {https://github.com/EcoAcao-Brasil/tocantins-framework}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed by [EcoAção Brasil](https://ecoacaobrasil.org) to support climate resilience research and urban planning initiatives.

## Support

- **Issues**: https://github.com/EcoAcao-Brasil/tocantins-framework/issues
- **Email**: isaque@ecoacaobrasil.org

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**Keywords**: urban heat island, thermal anomaly detection, Landsat, remote sensing, machine learning, spatial analysis, climate adaptation
