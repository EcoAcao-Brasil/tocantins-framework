"""
Basic Landsat 8/9 Analysis Example

This example demonstrates the simplest way to analyze Landsat 8/9 imagery
using default settings.
"""

from tocantins_framework import calculate_tocantins_framework

# Path to your Landsat 8/9 Level-2 Collection 2 GeoTIFF
TIF_PATH = "data/LC08_L2SP_example.tif"

# Output directory for results
OUTPUT_DIR = "results/landsat8_basic"

# Run complete analysis with default settings
# This works automatically for Landsat 8/9 scenes
calculator = calculate_tocantins_framework(
    tif_path=TIF_PATH,
    output_dir=OUTPUT_DIR
)

# Access results
feature_set = calculator.get_feature_set()
impact_scores = calculator.get_impact_scores()
severity_scores = calculator.get_severity_scores()

# Print summary
print(f"\n{'='*60}")
print("ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Total anomalies detected: {len(feature_set)}")
print(f"\nTop 5 anomalies by Impact Score:")
print(feature_set.nlargest(5, 'IS')[['Anomaly_ID', 'Type', 'IS', 'SS']])
print(f"\nResults saved to: {OUTPUT_DIR}/")
print(f"  - impact_scores.csv")
print(f"  - severity_scores.csv")
print(f"  - ml_features.csv")
print(f"  - anomaly_classification.tif")
print(f"  - lst_residuals.tif")
print(f"{'='*60}\n")
