# A_federated_multimodal_data-fusion_system_for_flood_detection_using_Federated_Learning
TriModalFloodNet: Multi-Modal Flood Prediction System
TriModalFloodNet is an advanced deep learning framework designed to predict flood events by fusing three distinct types of data: visual imagery, meteorological trends, and hydrological patterns. By combining "eyes," "meteorology," and "hydrology," the system achieves a holistic understanding of environmental conditions to provide accurate binary classifications (Safe vs. Flood).

## Architecture Overview
The system utilizes a tri-modal architecture where specific neural network backbones are assigned to each data stream.

1. The Eyes: Convolutional Neural Network (CNN)
Input: Satellite imagery (64x64 pixels) processed via Normalized Difference Water Index (NDWI) to highlight standing water.

Role: Extracts Visual Spatial Features, identifying current land cover and water boundaries.

2. The Meteorologist: TSMixer (Time-Series Mixer)
Input: 10 days of historical rainfall data.

Role: Extracts Rain Trend Features. It analyzes temporal patterns in precipitation to determine the intensity and duration of recent weather events.

3. The Hydrologist: NARX (Nonlinear AutoRegressive with eXogenous inputs)
Input: 20 hours of water level data (endogenous) and rainfall history (exogenous).

Role: Extracts Hydrological Features. It models the physical response of water bodies to external stimuli, predicting how levels will shift.

## The Fusion Process
Once the features are extracted from the three independent branches, they are integrated through a Fusion Layer:

Feature Concatenation: The high-level feature vectors from the CNN, TSMixer, and NARX are merged into a single comprehensive representation.

Classifier: The fused data is passed through a final classification head using a Sigmoid Activation function.

Output: A probability score that categorizes the situation as either SAFE (No immediate flood threat) or FLOOD (Imminent flood detected).

## Getting Started
Prerequisites
Python 3.8+

TensorFlow / PyTorch

OpenCV (for image processing)

Pandas/NumPy (for time-series data)

Data Preparation
Imagery: Place satellite TIFF or JPG files in the /data/images directory. Ensure they are pre-processed to 64x64 resolution.

Rainfall: Provide CSV files with daily precipitation totals for the last 10 days.

Water Levels: Provide hourly gauge readings for the preceding 20 hours.

## Why Tri-Modal?
Single-source models often fail due to:

Occlusion: Clouds blocking satellite views (solved by NARX/TSMixer).

Lag: Rainfall doesn't cause immediate floods (solved by CNN/NARX).

Complexity: Water levels vary by geography (solved by CNN/TSMixer context).

By merging these three perspectives, TriModalFloodNet provides a robust early warning system that is more resilient to sensor failure or environmental noise.
