<<<<<<< HEAD
# Industrial Embodied Intelligence Benchmarking Framework: Multiview Defect Detection

As industrial manufacturing becomes increasingly automated, the need for intelligent, adaptable, and perception-rich robotic systems is growing rapidly. Traditional inspection and automation systems, reliant on single-view analysis or rigid control algorithms, struggle in environments with complex geometries, dynamic lighting, and unpredictable defects.

This project aims to build an **Embodied Intelligence (EI) Benchmarking Suite** based on the **KubeEdge-Ianvs framework**, providing realistic datasets, simulation environments, and benchmarking pipelines that reflect real-world constraints in factories. By leveraging multi-view perception, SLAM-driven navigation, and robotics-centric datasets, this suite will accelerate the development and evaluation of embodied AI agents for manufacturing.

## Issues with Current Systems

1. **Lack of Multi-View Defect Detection Benchmarks**  
   Existing datasets often represent components from a single angle, failing to mimic the real-world dynamics of robotic inspection.

2. **Absence of Navigation + Perception Integration**  
   Current benchmarks rarely combine SLAM (Simultaneous Localization and Mapping) with visual inspection, a requirement in complex factory environments.

3. **Limited Dataset Flexibility**  
   Datasets lack modularity for multi-view, multi-sensor, and multi-task robotic workflows, hindering scalable benchmarking.

4. **Insufficient Support for Industrial-Specific Embodied AI Tasks**  
   Most embodied AI benchmarks focus on domestic or research lab tasks, not factory-floor challenges like bin picking, defect detection, and warehouse navigation.

## Goals

1. Build a multi-view industrial defect detection dataset (IISD) for robust perception benchmarking.
2.  Develop an industrial SLAM navigation dataset combining mapping, localization, and control.
3. Curate embodied AI and manipulation datasets relevant to industrial tasks.
4. Compile a list of key related research works supporting industrial embodied intelligence.
5. Develop a single-task learning example for defect detection using the IISD dataset in Ianvs.
6. Build a comprehensive benchmarking suite in Ianvs with standardized metrics and reporting.

## Design Details

### Dataset Map
### 📊 Industrial Surface Defect Detection Datasets

| Dataset Name | Description | Domain | Download Link |
|---------------|-------------|--------|----------------|
| **ISDD – Industrial Surface Defect Detection** | Multi-view defect detection dataset for gears, screws, nuts with 5 viewpoints using MANTA as the base. Includes defect types like scratches, stains, contamination, and indentations. **(Created by Ansh for LFX 2025 Pre-Test Submission)** | Industrial Defect Detection | [Link](https://drive.google.com/drive/u/1/folders/12JERdTIy_3WWRyjP2gm040TDYnmRxrxy) |
| **Real-IAD**                            | 150,000 images across 30 components with 5 views each for multi-view industrial anomaly detection. | Industrial Anomaly Detection       | [Link](https://realiad4ad.github.io/Real-IAD/)        |
| **NEU-CLS** | 1,800 grayscale images of hot-rolled steel with 6 defect types | Steel Surface | [Link](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data) |
| **Severstal: Steel Defect Detection** | 12,568 labeled images with pixel-wise annotations for steel defects | Steel Surface | [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection) |
| **ELPV** | 2,624 EL images of solar cells for defect detection | Solar Panels | [Link](https://github.com/zae-bayern/elpv-dataset) |
| **KolektorSDD** | 399 images (52 defective, 347 non-defective) of electrical commutators | Metal Surface | [Link](https://www.vicos.si/Downloads/KolektorSDD) |
| **DeepPCB** | PCB defect dataset with image pairs (template + defect) | PCB Inspection | [Link](https://github.com/Charmve/Surface-Defect-Detection/tree/master/DeepPCB)
| **Tianchi Aluminium Profile Dataset** | 10,000 images of aluminium profiles with defects like scratches and cracks | Aluminium | [Link](https://tianchi.aliyun.com/competition/entrance/231682/information) |
| **DAGM 2007** | 10 datasets for weakly-supervised defect detection on textures | Optical Inspection | [Link](https://zenodo.org/records/12750201) |
| **CrackForest** | Road surface and bridge crack dataset | Infrastructure | [Link](https://github.com/cuilimeng/CrackForest-dataset) |
| **Magnetic Tile Dataset** | Defects on magnetic tile surfaces with pixel-level annotations | Magnetic Tiles | [Link](https://github.com/Charmve/Surface-Defect-Detection/tree/master/Magnetic-Tile-Defect) |
| **RSDDs** | Rail surface defect dataset with noisy backgrounds | Railways | [Link](https://pan.baidu.com/share/init?surl=svsnqL0r1kasVDNjppkEwg) password: **nanr** |
| **KTH-TIPS** | Repeatable background texture dataset | Texture | [Link](https://pan.baidu.com/s/173h8V66yRmtVo5rc2P7J4A) |
| **Escalator Step Dataset** | Escalator step surface defect detection | Mechanical | [Link](https://aistudio.baidu.com/aistudio/datasetdetail/44820) |
| **Transmission Line Insulator Dataset** | Drone images of insulators with 600 normal and 248 defective | Power Lines | [Link](https://github.com/InsulatorData/InsulatorDataSet) |
| **MVTEC ITODD** | 3D object detection dataset for industrial applications | Industrial Objects | [Link](https://www.mvtec.com/company/research/datasets/mvtec-itodd) |
| **BSData** | 1,104 images with annotations for pitting defect progression | Machinery Wear | [Link](https://github.com/2Obe/BSData) |
| **GID - Gear Inspection Dataset** | 2,000 grayscale images with 28,575 defect annotations on gears | Mechanical | [Link](http://www.aiinnovation.com.cn/#/dataDetail?id=34) |


### 📊 Embodied Robotics and Manipulation

| Dataset Name                           | Description                                                                                                    | Domain                             | Link/Source       |
|-----------------------------------------|----------------------------------------------------------------------------------------------------------------|------------------------------------|-------------------|
| **SLAM Navigation Dataset**             | ROS2 & Gazebo-based dataset for warehouse, maze, and industrial navigation. Includes /scan, /odom, /map, /cmd_vel topics for benchmarking SLAM and navigation tasks. **(Created by Ansh for LFX 2025 Pre-Test Submission)**| Industrial Navigation/SLAM         | [Link](https://drive.google.com/drive/u/1/folders/1TpylmhkGnLNhCFw0fNS8wP-4pOKvlKqO) |
| **ARIO (All Robots In One)**            | Dataset covering 20+ real and simulated robots for tasks including locomotion, tool use, manipulation, and transport. | General Robotics/Embodied Learning |  [Link _[2024]_](https://imaei.github.io/project_pages/ario/https://imaei.github.io/project_pages/ario/)         |
| **Open X-Embodiment**                   | Aggregates 500+ robotic datasets into a common format for foundation models. Visual-language-action triplets included. | Robotic Foundation Models          | [Link _[2023]_](https://robotics-transformer-x.github.io/) |
| **RH20T-P**                             | Robotic hand dataset focused on primitive fine motor skills like flipping, insertion, pushing, with rich visual and force data. | Robotic Manipulation/Assembly      | [Link _[2024]_](https://sites.google.com/view/rh20t-primitive/main)         |
| **ALOHA 2**                             | Dual-arm coordination dataset with stacking, folding, and complex bimanual tasks. Enhanced RGB-D alignment. | Industrial Manipulation            | [Link _[2024]_](https://aloha-2.github.io/)|
| **Baxter_UR5_95_Objects_Dataset**       | RGB-D images and grasp annotations for 95 industrial objects using Baxter and UR5 robots. | Bin Picking, Warehouse Automation  | [Link _[2023]_](https://www.eecs.tufts.edu/~gtatiya/pages/2022/Baxter_UR5_95_Objects_Dataset.html) 
| **ISP-AD**                              | Industrial screen printing defect dataset with ink smudges, missing prints, ghosting errors. | Electronics/Screen Printing        | [Link](https://paperswithcode.com/dataset/isp-ad)|

## 🗺️ Why SLAM Datasets Matter

SLAM datasets are key for training and benchmarking robots that navigate industrial spaces like warehouses and factory floors. Robots need to localize themselves, map their environment, and move safely around obstacles and machinery.

### 🤔 How We Use SLAM Datasets

SLAM datasets capture real sensor values during navigation, including:

- **/scan** :  LIDAR scans for obstacle detection.
- **/odom** : Odometry for tracking movement.
- **/map** : Occupancy grid maps for spatial awareness.
- **/tf** : Spatial transforms between robot frames.
- **/cmd_vel** : Velocity commands for robot movement.

These datasets help test how well robots can build accurate maps, follow paths, and maintain precise localization over time, all critical for real-world industrial tasks like inspection, material transport, and autonomous navigation.



## ⛓️‍💥 Related Works: Embodied Intelligence Benchmarking in Industrial Manufacturing

| Title                                                                                   | Summary                                                                                                                       | Year |
|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|------|
| **A Survey of Embodied Learning for Object-Centric Robotic Manipulation**              | Explores passive to interaction-based learning for robotics, emphasizing self-supervised methods and long-horizon planning.   | [2022](https://arxiv.org/abs/2408.11537) |
| **Real-IAD: A Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection** | Benchmark dataset for multi-view industrial anomaly detection with unsupervised learning focus.                               | [2023](https://realiad4ad.github.io/Real-IAD/) |
| **RAD: Real-Life Anomaly Detection with Robotic Observations**                         | Dataset using robotic arms to collect multi-view anomaly data including RGB, point cloud, and 3D reconstructions.             | [2023](https://arxiv.org/abs/2406.07176) |
| **All Robots in One (ARiO)**                                                           | A unified API and dataset for 20+ robots across locomotion, manipulation, and tool use tasks.                                | [2024](https://arxiv.org/pdf/2408.10899) |
| **3CAD: A Large-Scale Real-World 3C Product Dataset for Anomaly Detection**             | Dataset with 27,000 annotated images for 3C (Computer, Communication, Consumer Electronics) manufacturing defects.            | [2024](https://arxiv.org/abs/2502.05761) |
| **voraus-AD: Anomaly Detection in Robot Applications with Time-Series and Telemetry**  | Focuses on robot-internal anomaly detection using telemetry like motor torques and actuator signals.                         | [2024](https://arxiv.org/abs/2311.04765) |
| **Embodied Intelligence Toward Future Smart Manufacturing in the Era of AI Foundation Model** | Discusses how large AI foundation models integrate into smart manufacturing with vision-language-action models.               | [2023](https://www.researchgate.net/publication/384417402_Embodied_Intelligence_Toward_Future_Smart_Manufacturing_in_the_Era_of_AI_Foundation_Model) |
| **Digital Twins to Embodied AI: Review and Perspective**                               | Explores the fusion of digital twins with embodied AI for real-time simulation, synchronization, and training.                | [2023](https://www.oaepublish.com/articles/ir.2025.11) |
| **A Comprehensive Survey on Embodied AI**                                              | Exhaustive review of embodied AI across perception, learning, simulators, and policy learning with recommendations.           | [2022](https://arxiv.org/html/2407.06886v1) |
| **EmbodiedScan: A Holistic Multi-Modal 3D Perception Suite**                           | A dataset and suite for 3D perception combining RGB, depth, surface normals, and point clouds for embodied agents.            | [2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_EmbodiedScan_A_Holistic_Multi-Modal_3D_Perception_Suite_Towards_Embodied_AI_CVPR_2024_paper.pdf) |
| **BEHAVIOR Vision Suite**                                                              | Synthetic dataset generation tool for task-based perception with adjustable lighting, objects, and camera trajectories.       | [2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ge_BEHAVIOR_Vision_Suite_Customizable_Dataset_Generation_via_Simulation_CVPR_2024_paper.pdf) |

## 🔍 Spotlight: Real-IAD Dataset - Multi-View Industrial Anomaly Detection

The **Real-IAD (Real-World Multi-View Dataset for Benchmarking Versatile Industrial Anomaly Detection)** dataset provides multi-view industrial anomaly detection capabilities. This dataset addresses the need for multi-view perception in manufacturing environments.

![Real-IAD Multi-View Dataset](assets/RealIAD.png)

*Figure: Real-IAD dataset showcasing multi-view anomaly detection across different industrial components. Each component is captured from 5 distinct viewpoints to simulate realistic robotic inspection scenarios.*

### 📊 Dataset Overview

**Real-IAD** contains **150,000 images** across **30 industrial components**, with each component captured from **5 different viewpoints**. This multi-view approach supports embodied intelligence systems that need to inspect objects from various angles, similar to robotic inspection scenarios.

**Note**: Access to this dataset has been requested using my institute ID and has been granted, making it available for our benchmarking framework development. This dataset is similar in concept to the ISDD (Industrial Surface Defect Detection) dataset I curated in my pre-test report, which also focuses on multi-view defect detection for industrial components.

### 🎯 Key Features

- **Multi-View Coverage**: Each component photographed from 5 viewpoints (0°, 72°, 144°, 216°, 288°)
- **Industrial Diversity**: 30 components from automotive, electronics, and mechanical sectors
- **Real-World Conditions**: Images captured under realistic lighting and environmental factors
- **Anomaly Types**: Covers defects including scratches, dents, contamination, and structural anomalies
- **Unsupervised Learning Focus**: Designed for unsupervised anomaly detection algorithms

### 🔧 Technical Specifications

| Metric | Value |
|--------|-------|
| Total Images | 150,000 |
| Components | 30 |
| Views per Component | 5 |
| Image Resolution | High-definition |
| Annotation Type | Pixel-level anomaly masks |
| Use Case | Unsupervised anomaly detection |

=======
version https://git-lfs.github.com/spec/v1
oid sha256:4134170e45ce14b6e3cfff7bcc2d8d78a51ed864df4679fc38c8d1ed0e912c35
size 15065
>>>>>>> 9676c3e (ya toh aar ya toh par)
