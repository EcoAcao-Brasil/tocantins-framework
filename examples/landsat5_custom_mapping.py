"""
Landsat 5 Analysis with Custom Band Mapping

This example shows how to analyze Landsat 5 imagery by providing
a custom band mapping configuration.
"""

from tocantins_framework import calculate_tocantins_framework

# Landsat 5 TM band mapping
# Note: Band names differ from Landsat 8/9
L5_BAND_MAPPING = {
    'blue': 'SR_B1',      # Blue (0.45-0.52 μm)
    'green': 'SR_B2',     # Green (0.52-0.60 μm)
    'red': 'SR_B3',       # Red (0.63-0.69 μm)
    'nir': 'SR_B4',       # Near-Infrared (0.77-0.90 μm)
    'swir1': 'SR_B5',     # SWIR1 (1.55-1.75 μm)
    'swir2': 'SR_B7',     # SWIR2 (2.08-2.35 μm)
    'thermal': 'ST_B6',   # Thermal (10.4-12.5 μm) - DIFFERENT FROM L8!
    'qa_pixel': 'QA_PIXEL'
}

# Path to Landsat 5 scene
TIF_PATH = "data/LT05_L2SP_example.tif"
OUTPUT_DIR = "results/landsat5_analysis"

# Run analysis with custom band mapping
calculator = calculate_tocantins_framework(
    tif_path=TIF_PATH,
    band_mapping=L5_BAND_MAPPING,
    output_dir=OUTPUT_DIR
)

# Access and display results
feature_set = calculator.get_feature_set()

print(f"\n{'='*60}")
print("LANDSAT 5 ANALYSIS COMPLETE")
print(f"{'='*60}")
print(f"Detected {len(feature_set)} thermal anomalies")

# Hot anomalies
hot_anomalies = feature_set[feature_set['Type'] == 'hot']
print(f"\nHot anomalies: {len(hot_anomalies)}")
if len(hot_anomalies) > 0:
    print(f"  Mean Impact Score: {hot_anomalies['IS'].mean():.3f}")
    print(f"  Mean Severity Score: {hot_anomalies['SS'].mean():.3f}")

# Cold anomalies
cold_anomalies = feature_set[feature_set['Type'] == 'cold']
print(f"\nCold anomalies: {len(cold_anomalies)}")
if len(cold_anomalies) > 0:
    print(f"  Mean Impact Score: {cold_anomalies['IS'].mean():.3f}")
    print(f"  Mean Severity Score: {cold_anomalies['SS'].mean():.3f}")

print(f"\nResults saved to: {OUTPUT_DIR}/")
print(f"{'='*60}\n")
