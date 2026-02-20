# Intelligent Deep Learning Video Surveillance System (Research Prototype)

## Abstract
This project presents an end-to-end intelligent video surveillance prototype. Designed as a highly modular pipeline, it fuses deep learning multi-object tracking (YOLOv8 + ByteTrack) with heuristic temporal behavior analysis to autonomously detect anomalies like loitering, running, and spontaneous crowd formation. The system aggregates these behavioral markers via a Decision Fusion Risk Engine to dynamically score targets in real-time.

## System Architecture

The software follows a strictly Object-Oriented, loosely-coupled design:
1. **Object Tracker**: Leverages `ultralytics` YOLOv8n coupled with BoT-SORT/ByteTrack. Chosen over pure DeepSORT because it offers highly optimized CPU inference times while maintaining SOTA ReID capabilities.
2. **Sub-Detector (Face Module)**: Uses Cascade Classifiers on dynamically generated Region of Interests (ROIs). This localized search dramatically reduces compute vs running inference over the entire frame.
3. **Behavior Analyzer**: A temporal buffer retains centroid history for $N$ frames per tracked ID. Using kinematic calculations (Euler distances / temporal variance) and graph-based proximity logic, it recognizes anomalous events.
4. **Risk Engine**: Fuses spatial (crowd) and temporal (speed/variance) indicators into an intuitive 0–100 risk score, incorporating automatic temporal decay when subjects return to baseline behavior.

## Installation & Usage

### Prerequisites
```bash
pip install opencv-python numpy ultralytics pyyaml