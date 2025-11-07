# Tocantins Framework

A comprehensive Python framework for detecting and quantifying intra-urban thermal anomalies in Landsat imagery using machine learning and spatial analysis techniques.

## Overview

The Tocantins Framework provides researchers and practitioners with robust tools for analyzing urban heat islands and thermal anomalies using Landsat 8/9 satellite imagery. The framework implements a scientifically rigorous methodology combining statistical analysis, machine learning, and spatial morphology to produce two complementary metrics:

- **Impact Score (IS)**: Quantifies the spatial extent and thermal significance of Extended Anomaly Zones (EAZs)
- **Severity Score (SS)**: Quantifies the thermal intensity of anomaly cores

These metrics enable data-driven urban heat intervention planning and climate adaptation strategies.

## Key Features

- **Automated Processing Pipeline**: End-to-end analysis from raw Landsat imagery to actionable metrics
- **Machine Learning Integration**: Random Forest-based residual analysis for robust anomaly detection
- **Spatial Coherence**: Morphological operations ensure spatially meaningful anomaly delineation
- **Standardized Metrics**: Reproducible, comparable thermal anomaly quantification
- **Flexible Configuration**: Customizable parameters for different urban contexts
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

### Basic Usage

```python
from tocantins_framework import calculate_tocantins_framework

# Run complete analysis on Landsat imagery
calculator = calculate_tocantins_framework(
    tif_path="path/to/landsat_image.tif",
    output_dir="results"
)

# Access results
feature_set = calculator.get_feature_set()
impact_scores = calculator.get_impact_scores()
severity_scores = calculator.get_severity_scores()
```

### Advanced Usage

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
    tif_path="path/to/landsat_image.tif",
    output_dir="results",
    save_results=True
)

if success:
    # Access intermediate results
    residual_map = calculator.get_residual_map()
    classification_map = calculator.get_classification_map()
```

## Input Data Requirements

The framework expects Landsat 8/9 Level-2 Collection 2 GeoTIFF files with the following bands:

- **SR_B1-B7**: Surface Reflectance bands
- **ST_B10**: Thermal band (Surface Temperature)
- **QA_PIXEL**: Quality Assessment band
- Additional QA and metadata bands

Data can be obtained from:
- [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- [Google Earth Engine](https://earthengine.google.com/)

## Output Files

The framework generates several output files:

### CSV Files
- `ml_features.csv`: Complete feature set with IS and SS metrics
- `impact_scores.csv`: Simplified Impact Scores
- `impact_scores_detailed.csv`: Detailed Impact Score breakdown
- `severity_scores.csv`: Core-level Severity Scores

### GeoTIFF Files
- `anomaly_classification.tif`: Spatial classification map (0=background, 1=cold EAZ, 2=hot EAZ, 3=cold core, 4=hot core)
- `lst_residuals.tif`: Land Surface Temperature residuals

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

### Scientific Foundation

The framework implements methodologies described in:

> [Paper citation still to be added after publication]

Key parameters are validated through empirical testing and sensitivity analysis.

## Configuration Parameters

### Spatial Parameters

```python
spatial_params = {
    'min_anomaly_size': 1,          # Minimum pixels for valid anomaly
    'agglutination_distance': 4,    # Dilation radius for core merging
    'morphology_kernel_size': 3,    # Morphological operation kernel size
    'connectivity': 2                # Pixel connectivity (1=4-connected, 2=8-connected)
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

### Development Setup

```bash
git clone https://github.com/EcoAcao-Brasil/tocantins-framework
cd tocantins-framework
pip install -e ".[dev]"
pytest tests/
```

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

- **Email**: isaque@ecoacaobrasil.org

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**Keywords**: urban heat island, thermal anomaly detection, Landsat, remote sensing, machine learning, spatial analysis, climate adaptation