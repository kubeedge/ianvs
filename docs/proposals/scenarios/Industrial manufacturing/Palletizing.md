# Embodied Intelligence Benchmarking Framework for Industrial Manufacturing with KubeEdge

With the accelerated evolution of intelligent industrial manufacturing, industrial robots, flexible production lines, and intelligent testing equipment are continuously being innovated. With its efficient data processing and real-time response capabilities, **cloud-edge collaboration technology** has become the core technology engine driving the application of **embodied intelligence** systems in complex industrial scenarios.  

Based on the **kubeedge ianvs collaborative AI benchmarking framework**, this project plans to build a complete system including industrial scenario test datasets, simulation test environment and multi-dimensional performance indicators, aiming to build professional and industry-level specific intelligent testing capabilities suitable for the industrial manufacturing field.  

It is noteworthy that the current industrial demand for embodied intelligent services has developed in depth in the direction of **high-precision perception decision-making, dynamic real-time response, cross device collaborative control, etc**.   

**However**, existing universal embodied intelligence benchmarks struggle to provide accurate assessments because they **lack targeted consideration of the unique needs of industrial scenarios and equipment characteristics**. This has become a key bottleneck restricting the development of industrial embodied intelligence technology.

## **Goals**

1. Introduce industrial datasets in the field of embodied intelligence, classify and reorganize the existing datasets according to four categories of standardized tasks: perception (such as multi view surface quality inspection), movement (such as material box handling, stacking, disassembling and stacking), operation (such as precision assembly, precision insertion, multi process flexible assembly, multi category flexible assembly, reel transfer, equipment in box packaging, mixed goods sorting), composition (such as electronic equipment flexibility test, power patrol inspection), and output the dataset Research Report.

2. Select at least one of the above scenarios in kubeedge ianvs to provide a standardized test suite, including datasets, test environments, and test indicators, and sort out the datasets in a standardized and unified data format.

3. Implement embodied intelligent baseline algorithm based on standardized test suite in kubeedge ianvs.

## **Design Details**

### **Dataset Map**

| Dataset | Scenarios | Introduction | Link |
| :-----| :----- | :----- | :----: |
| Agibot World | Operation/Perception/Composite | The AgiBot World dataset was born from Zhiyuan's self built large-scale data collection factory and application experimental base, with a total space area of over 4000 square meters and containing more than 3000 real objects. On the one hand, it provides a venue for large-scale data training of robots, and on the other hand, it realistically replicates the five core scenarios of home, catering, industry, supermarkets, and office, fully covering the typical application needs of robots in production and life. | [Link](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) |
| ARIO (All Robots in One), 2024 | Operation/Locomotion | ARIO is a comprehensive benchmark dataset designed to unify robot data across different implementation examples and task types. It covers over 20 real and simulated robot platforms, including tasks ranging from basic movements to complex tool usage and manipulation. Each robot is equipped with sensors such as RGB cameras, IMUs, and joint encoders, allowing agents to promote and transfer skills between platforms. In industrial environments where multiple robots must collaborate, such as wheeled bases for transporting parts and arms for performing assembly, ARIO's structure reflects the heterogeneity of these systems and the need for unified intelligence among them. | [Link](https://imaei.github.io/project_pages/ario/) |
| Open X-Embodiment, 2023 | Operation/Locomotion | 1 million fragments x 22 types of robots, covering 500+skills (including industrial sorting and assembly). | [Link](https://github.com/google-deepmind/open_x_embodiment) |
| RH20T-P (Robotic Hands Dataset with Primitive Skills), 2024 | Operation | For industries that focus on micro assembly or small part operations, such as gears, screws, and PCBs, RH20T-P provides valuable data for teaching robots precise and accurate interaction. | [Link](https://sites.google.com/view/rh20t-primitive/main) |
| ALOHA 2, 2024 | Operation | ALOHA 2 extends the original ALOHA dataset to include more complex dual arm coordination tasks. With improved physical realism and aligned RGB-D visual flow, it simulates tasks such as stacking, folding, and object alignment. This dataset captures the complexity of manual operation, which is crucial in many packaging and palletizing environments. | [Link](https://aloha-2.github.io/) |
| Baxter_UR5_95_Objects_Dataset, 2023 | Operation | This dataset includes RGB-D images of 95 industrial and household objects captured using Baxter and UR5 robotic arms. It has 3D grasping annotations, segmentation masks, and object poses. The inclusion of real robots and industrial like parts makes it particularly valuable to train and manipulate models under physical reality conditions.Its diversity and settings simulate the practical picking, component sorting, and mastery plan challenges often faced in warehouse automation systems. | [Link](https://tufts.app.box.com/s/t4apxhxtti6tlo9p875xtp823rhx1x9i) |
| YCB Object and Model Set | Operation | The YCB object and model set was created by the Yale Carnegie Mellon Berkeley collaboration to provide benchmark testing for robot operation, prosthetic design, and rehabilitation research. This dataset contains 73 everyday objects with different shapes, sizes, textures, weights, and rigidity, as well as some widely used operational tests. The dataset provides high-resolution RGBD scans, physical properties, and geometric models of objects, making it easy to integrate into operational and planning software platforms. In addition, the dataset also includes a series of standard task protocol frameworks and example protocols, as well as experimental implementations aimed at quantitatively evaluating various operational methods including planning, learning, mechanical design, control, and more. | [Link](http://rll.eecs.berkeley.edu/ycb/) |
| Human-Robot Collaboration Dataset | Operation/Locomotion | This synthetic dataset aims to simulate real data in industrial or service robot environments where humans and robots work side by side. | [Link](https://www.kaggle.com/datasets/adilshamim8/humanrobot-collaborative-work-dataset) |
| SmartAssemblySim-V2 | Operation | SmartAssemblySim-V2 is a conceptual subset of the BridgeData V2 dataset designed to simulate target condition robot operation tasks related to industrial manufacturing. The tasks include inserting parts, placing objects, and interacting with drawers, all supported by RGB videos, status data, and target commands. | [Link](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/) |
| NEU Surface Defect Database | Perception | The NEU surface defect database is a widely used benchmark for evaluating surface defect classification in the steel industry. It includes 1800 grayscale images of hot-rolled steel strips, classified into six types of defects such as silver lines, inclusions, patches, rough surfaces, rolled oxide scales, and scratches. This dataset is compact and efficient, suitable for rapid prototyping and algorithm testing. It is commonly used to develop models in quality assurance systems, where rapid defect identification is crucial. | [Link](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database/data) |
| ISDD - Industrial Surface Defect Detection Dataset | Perception | This dataset addresses the challenge of multi view defect detection in small industrial components such as nuts, gears, and screws. It is built on the MANTA dataset and provides five different perspectives of images for each object instance. | [Link](https://drive.google.com/drive/folders/12JERdTIy_3WWRyjP2gm040TDYnmRxrxy?usp=sharing) |
| Severstal: Steel Defect Detection | Perception | The Severstal dataset provides a large number of 12568 grayscale images of steel surfaces and detailed pixel level annotations for four types of defects. It is designed specifically for Kaggle competitions, promoting the development of advanced segmentation algorithms that can not only identify the presence of defects but also accurately locate them. Its well annotated structure makes it particularly suitable for training deep learning models used on automated steel inspection lines where spatial accuracy is crucial. | [Link](https://www.kaggle.com/c/severstal-steel-defect-detection) |
| BSData: Ball Screw Surface Defect Dataset | Perception | BSData is a domain specific dataset that focuses on inspecting key components in ball screw drives - CNC machine tools and automation equipment. This dataset contains over 21000 RGB images and detailed annotations of pitting defects, supporting supervised training of classification and detection models. It has been applied to evaluate how computer vision systems recognize local and subtle surface anomalies, providing valuable benchmarks for predictive maintenance systems in industrial environments. | [Link](https://github.com/2Obe/BSData) |
| Kolektor Surface-Defect Dataset | Perception | KolektorSDD and its subsequent SDD2 are datasets composed of high-resolution images of electric commutators. Each image is annotated to indicate the presence of surface defects such as scratches, dents, and contamination. These datasets are primarily used for binary and multi class classification, known for simulating real-world industrial constraints such as data imbalance and subtle defect patterns. They support research on rapid detection solutions that require high sensitivity to the minimum deviation of component surfaces. | [Link](https://www.vicos.si/Downloads/KolektorSDD) |
| Gear Inspection Dataset (GID) | Perception | GID contains grayscale images of industrial gears with defect labels. This dataset supports object detection and semantic segmentation tasks in industrial machine vision applications. It is particularly useful for training models aimed at real-time evaluation of gear quality, such as during online inspections on automotive or mechanical assembly lines. The focusing range of gears makes them an ideal choice for researching specific part defect detection in high-precision environments. | [Link](https://drive.google.com/file/d/1CZo-Ab5BXkTjV-b1-NIFzYMjfJQMl4nG/view?usp=share_link) |
| Real-IAD: Real-world Multi-View Industrial Anomaly Detection Dataset | Perception | Real IAD is a comprehensive benchmark designed to evaluate anomaly detection systems in real-world industrial environments. It includes 150000 high-resolution images of 30 different types of components, each taken from five different viewpoints. This dataset covers eight common types of defects, including cracks, dents, contamination, and misalignment. The design of Real IAD considers unsupervised anomaly detection, reflecting the challenges of high-throughput production lines where anomalies are rare but crucial for accurate identification. It supports research on anomaly localization, detection robustness, and multi view detection strategies, particularly suitable for quality assurance pipelines. | [Link](https://realiad4ad.github.io/Real-IAD/) |
| ISP-AD: Industrial Screen Printing Anomaly Detection Dataset | Perception | ISP-AD is a domain specific dataset for anomaly detection in screen printing processes. It includes examples of synthetic and real-world defects such as ink stains, missing prints, ghosting, and registration errors. The structure of this dataset supports classification and pixel segmentation, making it suitable for various industrial defect detection pipelines. Its main application scenarios include automatic visual inspection in the production of printed electronic products, labels, and packaging, ensuring consistent printing quality is crucial for product performance and brand. | [Link](https://paperswithcode.com/dataset/isp-ad) |
| MVTEC AD | Perception | The MVTec dataset contains 5354 high-resolution color images of different targets and texture types. It contains normal (i.e. defect free) images for training and abnormal images for testing. There are 70 different types of defects in anomalies, such as scratches, dents, contamination, and various structural changes. | [Link]( https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| RoboMIND | Composite | This dataset contains 107000 real-world demonstration trajectories involving 96 unique objects across 479 different tasks. The RoboMIND dataset collects operational data from various robot platforms, including 52926 Franka Emika Panda single arm robot trajectories, 19152 "Tiangong" humanoid robot trajectories, 10629 AgileX Cobot Magic V2.0 dual arm robot trajectories, and 25170 UR-5e single arm robot trajectory data. | [Link](https://data.flopsera.com/data-detail/21181956226031626?type=open) |

## **Ianvs**

The architectures and related concepts are shown in the below figure. The ianvs is designed to run within a single node. Critical components include  

&emsp;**Test Environment Manager:** the CRUD of test environments serving for global usage.  

&emsp;**Test Case Controller:** control the runtime behavior of test cases like instance generation and vanish.  

&emsp;&emsp;**Generation Assistant:** assist users to generate test cases based on certain rules or constraints, e.g., the range of parameters.  

&emsp;&emsp;**Simulation Controller:** control the simulation process of edge-cloud synergy AI, including the instance generation and vanishment of simulation containers.  

&emsp;**Story Manager:** the output management and presentation of the test case, e.g., leaderboards.

![Alt text](ianvs.png)

The current focus is on setting up the dataset in the Test Environment Manager section and the evaluation metrics section. At the same time, in the Test Case Controller section, use the Single task Learning Paradigm in Algorithm Paradigm to perform corresponding benchmark tests on the uploaded dataset.

## Construction of Palletizing Scenario

### Industrial Scenario: Palletizing

Palletizing is a key link in the industrial supply chain, connecting production and warehousing transportation. Its efficiency directly affects space utilization, logistics costs, and delivery speed, especially in industries such as food and medicine. Standardizing palletizing is the foundation for ensuring material safety. 

Robot simulation of palletizing has significantly improved compared to traditional manual/semi automated methods: 

&emsp;**1. Optimizing** paths through simulation rehearsals can increase efficiency by 3-5 times and can operate stably for 24 hours; 

&emsp;**2. Stacking** accuracy reaches ± 1mm, significantly reducing damage rate; 

&emsp;**3. Quickly** adapt to multiple types of materials, reduce production time by 80%, and better meet the needs of flexible production.

This scenario is built based on the RoboDK simulation environment, and an automated palletizing system is constructed that includes dual UR10 collaborative robots, conveyor belts, trays, and multi view virtual cameras. 

The dual UR10 robots (UR10 Base A and UR10 Base B) are responsible for palletizing tasks at different workstations, with conveyor belts (defined by motion logic based on frameworks such as ConveyorReference) serving as the workpiece transport carrier, pallets (PalletA and PalletB) used for storing and transferring workpieces, and multiple virtual cameras simulating visual inspection to achieve perception of workpiece position and posture, supporting precise robot operation.

The **real industrial scene palletizing** video is as follows:

![Alt text](Palletizing_real.png)


The video of **simulating industrial palletizing scene** in RoboDK is as follows:

![Alt text](Palletizing_sim.png)


From the comparison of the above two videos, it can be found that there are still shortcomings in the details of simulation compared to real industrial scenes, but it can largely restore the real industrial palletizing scene.

### The Overall Process of Palletizing Scenario

![Alt text](flow_chart.png)

**Installation position of cameras in industrial scenarios**

The three most commonly used camera installations in industrial scenarios are:
 
&emsp;**1. Eye to hand** gantry for high-speed stacking of single depth regular boxes; 

&emsp;**2. Eye in hand** is used for flexible grasping of soft bags/irregular parts; 

&emsp;**3. The hybrid implementation** of large field of view coarse positioning and end precision correction, balancing high speed and accuracy, has been validated on large-scale production lines.

In our palletizing simulation scenario, the camera uses a virtual **Eye-in-Hand** setup. It is installed on the robot’s end-effector and moves with it. Instead of relying on a fixed static reference frame, its detection logic is dynamically linked to the real-time pose of the end-effector. By continuously acquiring the end-effector’s pose, it calculates the workpiece's pose deviation relative to the camera via real-time coordinate transformations. This dynamic configuration aligns with physical Eye-in-Hand setups, adapting to varying viewing angles as the robot operates. It achieves high-speed positioning and palletizing of regular boxes, meeting industrial requirements for accuracy (±1mm) and cycle time (200-300 cycles per hour).  

## **Single Task Learning**

Single task learning is a traditional learning pooling all data together to train a single model. It typically includes a specialist model laser-focused on a single task and requires large amounts of task-specific labeled data, which is not always available on early stage of a distributed synergy AI project.  

As shown in the following figure, the single task learning works as procedures below:  

&emsp;1.Developer implements and deploys the application based on single task learning.  

&emsp;2.The application runs and launches single task learning.  

&emsp;3.The application uploads samples to the cloud.  

&emsp;4.Labeling service labels the uploaded samples.  

&emsp;5.Training learns the samples to generate a new model.  

&emsp;6.The system updates the model on the edge.  

&emsp;7.The model conducts inference given test samples where the inference result is sent to the application which ends the process.  

![Alt text](Single_Task_Learning.png)
    
The specific implementation of Palletizing single task learning algorithm in `algorithm.yaml`.

The URL address of the algorithm is filled in the configuration file `benchmarkingjob.yaml` .

### Algorithm

|Target/Object | Input Data | Common Industrial Algorithms|
| ----- | -------------- | --------------------- |
|Box positioning | 2D RGB/3D point cloud | YOLOv8 seg+3D minimum bounding box|
|Pose estimation | 3D point cloud | ICP registration/Pose CNN|
|Grasping point | Box posture+fixture geometry | GraspNet/torque balance|
|Joint denoising | Joint angle time series | Kalman filtering|
|Trajectory optimization | Joint angle | RRT*/TOPP|
|Abnormal detection | Joint angle+torque | LSTM/One Class SVM|

**YOLOv8-seg+3D minimum bounding box** is used for box positioning, **GraspNet** is used for grasping, **Kalman filter** is used for joint noise, **RRT\*** is used for paths, and **LSTM** is used for anomalies.

**YOLOv8** is the latest version of the YOLO (You Only Look Once) series object detection algorithm, which has improved both speed and accuracy. YOLOv8 introduces a new network structure and optimization strategy that supports multi-scale feature fusion and more efficient anchor management to improve detection performance. In addition, it enhances the detection capability for small targets and provides more flexible model deployment options, suitable for various real-time object detection scenarios.

In this project, we use YOLOv8 to perform **pure object detection** on the images captured by the camera.

![Alt text](YOLOv8_latest.png)

## RoboDK Palletizing Dataset

Due to RoboDK's output format being more inclined towards "raw data records within the scene", while Ianvs requires "standardized, structured, and correlatable test data", there is a high probability of differences between the two native formats. Therefore, it is necessary to convert or adapt the data format according to Ianvs' specifications to ensure that the data is correctly parsed and used for algorithm testing.

For the motion information of the robot and the image information of the camera obtained later, we can use the algorithm in the above table to further process them.

The ultimate dataset form:

```yaml
RoboDK Palletizing Dataset/
├─ robo_snapshots        # original images
│  ├─ 20250814_100620.png
│  ├─ 20250814_100720.png
│  ├─ ...
├─ robo_joints_log       # 6-axis joint angles
│  ├─ 20250814_100620.csv
│  ├─ 20250814_100720.csv
│  ├─ ...
├─ robo_joints_log_json  # TCP pose (xyz+quat or xyzrpw)
│  ├─ 20250814_100620.json
│  ├─ 20250814_100720.json
│  ├─ ...
├─ robo_depth_images           
│  ├─ 20250814_100620.tiff         # 32F depth
│  ├─ 20250814_100620_vis.png      # pseudo-color visualization
│  ├─ 20250814_100720.tiff         # 32F depth
│  ├─ 20250814_100720_vis.png      # pseudo-color visualization
│  ├─ ...

```

In the benchmark of this research plan, only the raw image data from **robo_snapshots** was used, while the rest of the data needs further follow-up research.


### Images Part

The image captured by the camera is roughly as shown in the following picture:

![Alt text](camera_capture.png)

### Robot Motion Information

Robot arm joint information and TCP pose information in CSV format, here is a test format:

```
# Robot State @ 2025-08-14 10:49:19

┌───────────────────────────────┐
│ Joints (degrees)              │
│ ┌─────┬───────┐               │
│ │ J1  │ -8.247│  Base rotation│
│ │ J2  │ -99.686│ Shoulder      │
│ │ J3  │ 80.867 │ Elbow         │
│ │ J4  │ -71.181│ Wrist 1       │
│ │ J5  │ -90.0  │ Wrist 2       │
│ │ J6  │ -8.247 │ Wrist 3       │
│ └─────┴───────┘               │
├───────────────────────────────┤
│ TCP Pose (mm / degrees)       │
│ ┌───────┬───────────┐         │
│ │ X     │ 309.964   │ mm      │
│ │ Y     │ 1839.792  │ mm      │
│ │ Z     │ 579.970   │ mm      │
│ │ Rx    │ -180.0    │ deg     │
│ │ Ry    │ -1.12E-05 │ deg     │
│ │ Rz    │ -1.00E-05 │ deg     │
│ └───────┴───────────┘         │
└───────────────────────────────┘
```

## Directory Structure: (examples/Palletizing)

```yaml
Palletizing
└── singletask_learning_bench
    ├── benchmarkingjob.yaml
    ├── testalgorithms
    │   ├── yolov8.py
    │   ├── algorithm.yaml
    └── testenv
        ├── metric.py
        └── testenv.yaml
    ├── readme.md
```

In this project, the selected metric is accuracy.


## **Road Map**

**1.** **July to Mid-August**  
Conduct research on currently available embodied intelligent datasets and produce corresponding reports. Simultaneously, continue to follow up on and refine the proposal. Additionally, learn to use the RoboDK platform and construct a palletizing scene within it.  

**2.** **Mid-August to Mid-September**  
Acquire the corresponding dataset. Establish the test environment and test indicators in KubeEdge Ianvs, and standardize and unify the dataset into a consistent data format. Concurrently, implement the specific intelligent baseline algorithm in KubeEdge Ianvs based on the standardized test suite.  

**3.** **Mid-September to End of September**  
Summarize the outcomes of the previous two stages, analyze areas for further improvement or supplementation, and produce corresponding documentation. If time and resources permit, extend the standardization of the test suite—including indicators and examples—to AgiBot World and Genie SIM, a smart metadata simulation platform.

## **Reference**

1. *Shenzhen Hengzhi Image Technology Co., Ltd A fixed bracket for industrial cameras: CN202021042021.7 [P]. January 8, 2021.*

2. *L Wang, H S Min. Dual Quaternion Hand eye Calibration Algorithm Based on LMI Optimization [J]. Machine Tool and Hydraulic, 2021, 49 (21): 8-14. DOI: 10.3969/j.issn.1001-3881.2021.002.*

3. *J C Guo, Z M Zhu, Y F Yu, etc. Research and Application of Laser Structured Light Vision Sensing Technology in Welding Field [J]. China Laser, 2017, 44 (12): 1-10. DOI: 10.3788/CJL201744.1200001.*
