# Edge Intelligence Benchmarking Framework: Multi-Object Tracking

As edge computing becomes increasingly prevalent, the need for intelligent, real-time, and perception-rich tracking systems is growing rapidly. Traditional tracking systems—reliant on single-node processing or cloud-only architectures—struggle in environments with network latency, bandwidth constraints, and resource limitations at the edge.

This project aims to build a **Multi-Object Tracking (MOT) Benchmarking Suite** based on the **KubeEdge-Ianvs framework**, providing realistic datasets, edge deployment configurations, and benchmarking pipelines that reflect real-world constraints in edge computing environments. By leveraging multi-edge inference, distributed processing, and tracking-centric evaluation metrics, this suite will accelerate the development and evaluation of edge AI systems for tracking applications.

***

## Goals

1. Build a multi-edge MOT17 benchmark for robust distributed tracking evaluation.
2. Develop edge-optimized tracking algorithms with resource-aware deployment.
3. Create standardized metrics for edge tracking performance including latency, throughput, and accuracy trade-offs.
4. Compile key research works supporting edge-based multi-object tracking.
5. Develop automated setup and execution pipelines for MOT17 edge benchmarking in Ianvs.
6. Build a comprehensive benchmarking suite with edge-specific evaluation metrics.

***

## Design Details

### Dataset Map

#### 📊 Multi-Object Tracking Datasets

| Dataset Name | Description | Domain | Download Link |
|--------------|-------------|--------|--------------|
| **MOT17** | 21 sequences (7 train + 14 test) with 3 detector variants (DPM, FRCNN, SDP) for pedestrian tracking. **Edge-optimized benchmark implementation in KubeEdge-Ianvs** | Pedestrian Tracking | [MOTChallenge](https://motchallenge.net/data/MOT17/), [Kaggle MOT17](https://www.kaggle.com/datasets/wenhoujinjust/mot-17), [KubeEdge MOT17 Dataset (Kaggle)](https://www.kaggle.com/datasets/nishantsinghhhhh/kubeedge-dataset-mot17/data?select=all_rank_reid_job.csv)[1] |

#### 📊 Edge Computing and Distributed AI

| Dataset Name         | Description                                                                          | Domain           | Link/Source                                                                                                                                       |
|----------------------|--------------------------------------------------------------------------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **MOT17-Edge Benchmark** | Multi-edge inference benchmark for MOT17 with GPU/CPU node configurations. **Built for KubeEdge-Ianvs framework** | Edge AI Tracking | [Ianvs Multi-Edge Inference Bench](https://github.com/kubeedge/ianvs/tree/main/examples/MOT17/multiedge_inference_bench)                          |
| **EdgeAI-Bench**     | Comprehensive edge AI benchmarking suite across multiple tasks                       | Edge Computing   | [EdgeAI-Bench](https://github.com/microsoft/EdgeML)                                                                                               |

***

## Why MOT17 Edge Benchmarks Matter

MOT17 edge benchmarks are crucial for evaluating how tracking algorithms perform under real-world edge constraints. Edge deployments face unique challenges including limited computational resources, network latency, and the need for real-time performance.

### How We Use MOT17 Edge Benchmarks

The MOT17 edge benchmark captures performance across multiple dimensions:

- **Multi-Edge Inference**: Distributing tracking across GPU and CPU edge nodes
- **Resource Utilization**: Memory usage, compute efficiency, and power consumption
- **Network Efficiency**: Bandwidth usage and latency tolerance
- **Real-Time Performance**: Frame rate, processing delay, and throughput metrics
- **Accuracy Trade-offs**: Tracking precision vs computational efficiency

These benchmarks help evaluate how well tracking algorithms maintain accuracy while meeting the strict resource and latency requirements of edge deployments.

***

## Dataset Structure for Benchmarking

A transparent schema for dataset, experiment, and reporting is essential for reproducibility:

| Dataset | Structure                                      | Time Paradigm      | Algorithm                | Base Model                | Batch Size | mAP   | Rank-1 | Rank-2 | Rank-5 | CMC   |
|---------|-----------------------------------------------|--------------------|--------------------------|---------------------------|------------|-------|--------|--------|--------|-------|
| MOT17   | 7 train/14 test; 3 detectors (DPM, FRCNN, SDP) | Real-Time, Offline | FairMOT, DeepSORT, ByteTrack, EdgeMOT, etc. | YOLO, CenterNet, Faster R-CNN (detectors); Custom Trackers | Variable (typically 1-16) | Reported per experiment | Reported if re-ID is evaluated | Reported if re-ID is evaluated | Reported if re-ID is evaluated | CMC curves commonly shown |

- **Dataset**: Name used, e.g., MOT17, MOT17-Edge
- **Structure**: Train/test split, number of detectors, scenes, annotation types
- **Time Paradigm**: Real-time or offline; edge prioritizes real-time
- **Algorithm**: The tracker under evaluation (FairMOT, ByteTrack, DeepSORT, EdgeMOT)
- **Base Model**: The backbone detection or tracking network
- **Batch Size**: Hardware-dependent, smaller for edge settings (up to 16)
- **mAP, Rank-1/2/5, CMC**: Standard MOT and re-identification metrics

#### Example Structure Row

| Dataset | Structure                     | Time Paradigm      | Algorithm        | Base Model | Batch Size | mAP   | Rank-1 | Rank-2 | Rank-5 | CMC   |
|---------|-------------------------------|--------------------|------------------|------------|------------|-------|--------|--------|--------|-------|
| MOT17   | 21 seqs, 3 detectors, street/indoor | Real-Time Inference | ByteTrack+YOLOv8 | YOLOv8     | 1         | 78.3  | 54.3   | 70.1   | 82.6   | Shown |

*Values for mAP, Rank-1/2/5/CMC depend on particular algorithm runs and experiment.*

### Edge Benchmarking Metrics and Organization

For edge computing experiments, always report:

- **Latency**: Time per frame, network transmission time, inference delay
- **Throughput**: Frames-per-second (FPS), concurrent streams
- **Accuracy**: mAP, ID metrics (IDF1, ID switches), CMC curves
- **Resource Usage**: Memory footprint, CPU/GPU utilization, power consumption
- **Bandwidth Usage**: Upstream/downstream traffic

Include:

- **Algorithm/detector combinations, batch size, edge hardware configuration, edge-specific constraints**
- **Real-time performance curves, CMC curves, and accuracy-resource tradeoff plots**

***

## ⛓️ Related Works: Multi-Object Tracking in Edge Computing

| Title | Summary | Year |
|-------|---------|------|
| **Real-time Multi-Object Tracking with Deep Learning on Edge Devices** | Lightweight tracking algorithms for edge hardware with YOLO-based detection and DeepSORT tracking. | [2023](https://arxiv.org/abs/2308.12169) |
| **Edge-Cloud Collaborative Multi-Object Tracking for Autonomous Driving** | Framework for distributing tracking tasks between edge devices and cloud servers for autonomous vehicles. | [2022](https://ieeexplore.ieee.org/document/9847329) |
| **FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking** | Joint detection and re-identification for real-time multi-object tracking. | [2020](https://arxiv.org/abs/2004.01888) |
| **ByteTrack: Multi-Object Tracking by Associating Every Detection Box** | State-of-the-art tracking with low computational overhead. | [2021](https://arxiv.org/abs/2110.06864) |
| **Towards Real-Time Multi-Object Tracking** | Survey of real-time MOT methods with focus on computational efficiency and deployment considerations. | [2022](https://arxiv.org/abs/2202.11773) |
| **EdgeMOT: Multi-Object Tracking at the Network Edge** | Distributed tracking for network edge deployments. | [2023](https://dl.acm.org/doi/10.1145/3583120.3586961) |
| **Federated Multi-Object Tracking with Privacy Preservation** | Collaborative tracking with privacy using federated learning across edge nodes. | [2023](https://arxiv.org/abs/2305.14386) |
| **Resource-Aware Multi-Object Tracking on Edge Devices** | Adaptive tracking based on available computational resources. | [2022](https://ieeexplore.ieee.org/document/9762582) |
| **MOTChallenge: A Benchmark for Single-Camera Multiple Object Tracking** | Overview of MOT metrics and benchmarks, including MOT17. | [2021](https://link.springer.com/article/10.1007/s11263-020-01393-0) |
| **Deep Learning for Multi-Object Tracking: A Survey** | Survey of deep learning approaches for MOT, efficiency, and deployment. | [2023](https://arxiv.org/abs/2301.04748) |

***

##  Spotlight: MOT17 - Multi-Detector Pedestrian Tracking Benchmark

The **MOT17 (Multiple Object Tracking Benchmark 2017)** dataset provides comprehensive pedestrian tracking evaluation with multiple detector variants. This dataset addresses the need for robust tracking evaluation across different detection qualities in real-world scenarios.



*Figure: MOT17 dataset showcasing multi-detector tracking results. Each sequence provides detection results from DPM, FRCNN, and SDP detectors, enabling comprehensive tracking algorithm evaluation.*

### Dataset Overview

**MOT17** contains **21 sequences** across **7 unique scenes**, with each scene providing detections from **3 different detectors** (DPM, FRCNN, SDP). This multi-detector approach enables robust evaluation of tracking algorithms under varying detection qualities.

#### Key Features

- **Multi-Detector Support**: Each sequence has DPM, FRCNN, and SDP detection variants
- **Diverse Scenarios**: Street scenes with varying crowd densities and lighting conditions
- **Real-World Conditions**: Captured in challenging urban environments with occlusions
- **Rich Annotations**: Frame-by-frame bounding boxes with consistent identity labels
- **Edge-Optimized Benchmark**: KubeEdge-Ianvs implementation for distributed edge inference

#### Technical Specifications

| Metric                    | Value                                     |
|---------------------------|-------------------------------------------|
| Total Sequences           | 21 (7 unique × 3 detectors)               |
| Training Sequences        | 7 unique scenes                           |
| Test Sequences            | 7 unique scenes                           |
| Total Frames              | ~11,200                                   |
| Average Objects per Frame | 5-30                                      |
| Frame Rate                | 30FPS                                     |
| Resolution                | 1920×1080 (varies)                        |

### KubeEdge-Ianvs Edge Implementation

The MOT17 benchmark in KubeEdge-Ianvs provides:

- **Multi-Edge Inference**: Distributed tracking across GPU and CPU nodes

This implementation makes MOT17 the first multi-object tracking benchmark specifically designed for edge computing evaluation, addressing the critical gap between traditional accuracy-focused benchmarks and real-world deployment requirements.

***

## Conclusion & Usage

Use the structure and metric schema outlined above (with added Kaggle dataset links, including the KubeEdge-MOT17 variant) to organize edge AI benchmarking experiments. Reporting results with standardized tables, real-time metrics, and edge-specific resource analysis maximizes reproducibility and impact for modern, distributed tracking research.[1]

[1] https://www.kaggle.com/datasets/nishantsinghhhhh/kubeedge-dataset-mot17/data?select=all_rank_reid_job.csv