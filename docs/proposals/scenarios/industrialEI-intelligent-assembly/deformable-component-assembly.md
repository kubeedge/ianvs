# Embodied Intelligence Industrial Assembly:Implementation on KubeEdge-lanvs 

---

## Introduction

With the evolution of intelligent industrial manufacturing, industrial robots, flexible production lines, and intelligent testing equipment are continuously being innovated. With its efficient data processing and real-time response capabilities, the need for intelligent, adaptable, and perception-rich robotic systems is also growing rapidly.

It is noteworthy that the current industrial demand for embodied intelligent services has developed in depth in the direction of high-precision perception decision-making, dynamic real-time response, and integrated multi-stage control.

This project aims to build an **Embodied Intelligence (EI) Benchmarking Suite** based on the **KubeEdge-Ianvs framework**, providing realistic datasets, simulation environments, and benchmarking pipelines that reflect real-world constraints in industries.Our goal is to build professional and industry-level specific intelligent capabilities suitable for the high-precision field of industrial assembly.

However, existing universal embodied intelligence benchmarks struggle to provide accurate assessments because they lack targeted consideration of complex, end-to-end assembly workflows that combine perception, force-control, and intelligent quality assurance. This has become a key bottleneck restricting the development of industrial embodied intelligence technology.

---

## Goals

1. **Build a multi-modal industrial assembly dataset**: Create a comprehensive, multi-modal dataset with URDFs, camera images, and force/torque sensor data to enable robust perception and manipulation benchmarking.

2.**Develop an end-to-end assembly pipeline**: Implement a full assembly workflow that goes from component detection and placement to quality inspection and final packaging.

3.**Integrate multi-task AI capabilities**: Combine object detection, force-controlled manipulation, and visual inspection into a single, cohesive algorithm.

4.**Create a conditional decision-making framework**: Develop the if-else logic for quality assurance and intelligent sorting, demonstrating the system's ability to adapt to its own performance.

5.**Compile a list of key related research works**: Provide a list of relevant external datasets and algorithms (like REASSEMBLE ,DeepPCB and YOLOv8) that support the project's technical approach.

6.**Build a comprehensive benchmarking suite**: Develop a benchmarking suite within ianvs that includes standardized metrics, detailed test reports, and a leaderboard for evaluating multi-stage assembly performance.

---

## Proposal

## Design Details


### **Dataset Map**
This project will create a custom dataset for a complete, end-to-end industrial assembly task. However, the following existing datasets are referenced to validate and benchmark individual sub-tasks of the project's multi-stage workflow.

| Dataset Name | Description | Domain | Link/Source |
| :--- | :--- | :--- | :--- |
| REASSEMBLE | A multimodal dataset for contact-rich robotic assembly and disassembly, including force-torque sensor data, multi-view images, and annotations. | Robotic Manipulation/Assembly | [Link](https://arxiv.org/html/2502.05086v1) |
| DeepPCB | A PCB defect dataset with image pairs (template + defect) that is ideal for training quality inspection models. | PCB Inspection | [Link](https://github.com/Charmve/Surface-Defect-Detection/tree/master/DeepPCB) |
| KolektorSDD | A dataset of images of electrical commutators with surface defects like scratches and cracks, useful for robust quality control. | Metal Surface | [Link](https://www.vicos.si/Downloads/KolektorSDD) |
| YCB Object and Model Set | A benchmark dataset containing 73 common objects with high-resolution RGB-D scans, physical properties, and geometric models. | Object Manipulation | [Link](http://rll.eecs.berkeley.edu/ycb/) |
| Open X-Embodiment | Aggregates 500+ robotic datasets into a common format for foundation models. Covers a wide range of skills including industrial sorting and assembly. | General Robotics/Embodied Learning | [Link](https://github.com/google-deepmind/open_x_embodiment) |
| SmartAssemblySim-V2 | A conceptual dataset subset for robot operation tasks like inserting and placing parts, with RGB videos and status data. | Industrial Manufacturing | [Link](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/) |
| ISDD - Industrial Surface Defect Detection Dataset | A multi-view defect detection dataset for small industrial components, providing five different perspectives of images for each object. | Industrial Defect Detection | [Link](https://drive.google.com/drive/folders/12JERdTIy_3WWRyjP2gm040TDYnmRxrxy?usp=sharing) |
| RH20T-P | A robotic hand dataset focused on primitive fine motor skills like flipping, insertion, and pushing, with rich visual and force data. | Robotic Manipulation/Assembly | [Link](https://sites.google.com/view/rh20t-primitive/main) |
| Human-Robot Collaboration Dataset | A synthetic dataset that simulates real data in environments where humans and robots work side-by-side. | Human-Robot Interaction | [Link](https://www.kaggle.com/datasets/adilshamim8/humanrobot-collaborative-work-dataset) |

---

### What is Deformable Component Assembly ?

Modern gadgets like smartphones and laptops contain motherboards with increasingly small and delicate components. Assembling these parts requires a level of precision and consistency that is difficult to achieve with manual labor. This is the core challenge of deformable component assembly.
[Image description](images/industrial_assembly.png)

### What do this dataset do ?

 This  project addresses this by using an embodied intelligent robotic arm. This robotic arm leverages a fusion of advanced capabilities, including **force detection** for delicate haptic control, **visual detection** for accurate positioning, and **data-driven intelligence** to store, compare, and verify each assembly step. By integrating these systems, the robot can not only assemble components with superior precision but also perform real-time quality checks to ensure a perfect final product.
[Image description](images/working_assembly_flowchart.png)

---

## Ianvs 

### Ianvs Framework Integration Architecture

Our pose estimation algorithm is designed to integrate seamlessly with the Ianvs framework through its three core components, as illustrated in the following architecture diagram:

![Ianvs Framework Architecture](assets/ianvs-architecture.drawio.svg)

*Figure above: Ianvs framework architecture showing the integration of Test Environment Manager, Test Case Controller, and Story Manager for benchmarking embodied intelligence in industrial environments.*

The initial focus is on configuring the custom, multi-modal `Intelligent Assembly Dataset` within the Test Environment Manager and establishing the multi-stage performance metrics. Simultaneously, the `Test Case Controller` will be implemented with a new Multi-task Assembly Paradigm. This paradigm will perform comprehensive benchmark tests on the custom dataset, evaluating the full, end-to-end workflow—from component positioning with `YOLOv8` to the final quality inspection using a CNN.

###  Directory Structure

The new dataset will be generated and uploaded in the https://www.kaggle.com/datasets/kubeedgeianvs ,users can install the zip file and then unzip in in their datasets folder .

```
──ianvs/examples/robot/lifelong__learning__bench/
              └── deformable__assembly/
                  ├── dataset/  
                  ├── testalgorithms/
                  │   └── assembly__alg/   
                  │       ├── __init__.py
                  │       ├── basemodel.py            # Core pose estimation algorithm
                  │       ├── data_utils.py           
                  │       ├── eval.py
                  │       ├── robot_control.py
                  │       ├── perception.py           # YOLO model for object detection and CNN for visual inspection
                  │       ├── decision_logic.py       # if-else logic for quality check and cable sorting
                  │       └── assembly_planner.py     #  Orchestrates the entire multi-stage assembly process
                  ├── testenv/
                  │   └── assembly__env/    
                  │       ├── __init__.py
                  │       ├── testenv.yaml
                  ├── utils/
                  │   ├── __pycache__/
                  │   └── view_dataset.py   # For viewing a local copy of the downloaded dataset
                  ├── .gitignore
                  ├── LICENSE
                  ├── README.md             # documentation for users
                  └── benchmarkingjob.yaml
```


### **Multi-task Assembly Paradigm**

The Multi-task Assembly Paradigm goes beyond traditional single-task learning by integrating multiple, sequential AI capabilities into a single cohesive workflow. This approach is more representative of a real-world, end-to-end industrial process and requires a more complex and robust model. It is designed to demonstrate an embodied intelligent system that can perceive, manipulate, verify, and reason.

The multi-task assembly paradigm works as follows:
1.  The developer creates and deploys an application based on a multi-task learning framework.
2.  The application runs and launches the full assembly workflow, which includes a series of interdependent tasks.
3.  The robotic arm uses **YOLOv8** to perform real-time object detection on components.
4.  The robot then executes a **force-controlled assembly** task to place the components on the motherboard.
5.  After assembly, the robot's camera captures an image for **visual inspection** and a **CNN-based model** performs a quality check against an ideal reference.
6.  An `if-else` decision-making process is executed: if the quality check passes, the workflow proceeds; otherwise, it stops.
7.  The robot performs a final logical task, sorting the correct cable (e.g., USB-C or USB-B) and packaging the final product.
8.  The system then generates a final test report and updates the leaderboard, concluding the process.


`The specific implementation of Palletizing single task learning algorithm in `algorithm.yaml`
`The URL address of the algorithm is filled in the configuration file `benchmarkingjob.yaml` .

[Image description](images/intelligent_multi-stage_assembly.png)


---

## Algorithm

### **Algorithm**

| Target/Object | Input Data | Common Industrial Algorithms |
| :--- | :--- | :--- |
| **Component Positioning** | Live RGB-D camera feed `(PyBullet)` | `YOLOv8` (for object detection), `Pose CNN` (for 6D pose estimation) |
| **Robotic Arm Trajectory & Grasping** | Object position ($x, y, z$), orientation, component URDFs | GraspNet, RRT* (for path planning), Kinematics/Inverse Kinematics |
| **Deformable Assembly** | Live force/torque sensor data, joint angles | Force/Impedance Control, Compliance Control |
| **Visual Inspection & Defect Detection** | RGB image of assembled motherboard | `Convolutional Neural Network (CNN)` for Image Classification or Anomaly Detection |
| **Logical Decision Making 1** | Result of visual inspection (Pass/Fail) | `if-else` control flow, Finite State Machine |
| **Cable Type Recognition** | Motherboard component features | Rule-based classifier, Computer Vision (OCR if serial numbers are present) |
| **Logical Decision Making 2** | Result of cable recognition (C-type or B-type) | `if-else` control flow, State Machine |
| **Final Packaging** | Phone pose, cable pose, box pose | GraspNet, Motion Planning |

---

## Quality Control and Decision Making:

### **Intelligent Decision-Making: The Cable Sorting Logic**

A core component of our multi-task paradigm is the implementation of intelligent, conditional logic. After a successful assembly and quality check, the robotic arm must decide which accessory to include with the finished product. This is based on an `if-else` condition tied to a specific component on the motherboard.

The process is as follows:
* The system uses its perception module to confirm which type of USB receptacle component has been successfully assembled on the motherboard.
* **Condition:** `IF` a USB-C receptacle component is detected on the motherboard, `THEN` the robotic arm selects a USB-C cable from the pallet.
* **Condition:** `ELSE IF` a micro USB (Type B) receptacle component is detected, `THEN` the robotic arm selects the micro USB cable.

This decision-making process demonstrates the robot's ability to reason, adapt its actions based on the specific assembly, and ensure the final package is complete and correct.
[Image description](images/cable-if-else-logic.png)

### **AI-Driven Quality Assurance & Benchmarking**

A cornerstone of our multi-task assembly paradigm is the self-verification of the robot's work, which provides a critical feedback loop for quality assurance. This process moves beyond simple object detection and into intelligent defect detection, serving as a primary benchmark for the `ianvs` framework.

The process is as follows:
* After the robotic arm completes the assembly of all components, it uses its onboard camera to capture a high-resolution image of the final product.
* This image is then processed by a **pre-trained CNN model** that performs a **visual inspection**. This model has been trained on a custom dataset to recognize an "ideal" assembly state.
* **Condition:** An `if-else` logical check is performed. `IF` the visual inspection model determines the current assembly matches the ideal state, `THEN` the result is logged as "Pass." `ELSE`, the result is logged as "Fail."
* The final outcome of this quality check—Pass or Fail—is captured by the `ianvs` framework and is a primary metric in the generated test reports and leaderboards. This mechanism directly evaluates the algorithm's ability to perform precise manipulation and provides a clear, objective measure of its success.
[Image description](images/CNN_of_motherboard.png)
---

## Dataset Management

Different datasets will be used to train the robotic AI model:

1.**The New Dataset** : It is the custom-made data from PyBullet simulation, including the URDFs, images of the assembled phone, and sensor logs.

2.**REASSEMBLE** :  Used as a "practice test" to train and validate the fundamental skills of your robotic arm, especially for force-controlled insertion.

3.**DeepPCB** : t's a collection of images of electronic circuit boards with defects. This data is used to train quality control AI to recognize defects on a motherboard.


### Dataset Structure 

```
├── assembly_dataset/
│   ├── stage1_component_detection/
│   │   ├── images/
│   │   │   ├── 001_panel_view.png
│   │   │   ├── 002_panel_view.png
│   │   │   └── ...
│   │   └── annotations/
│   │       ├── 001_panel_view.json      # YOLO/COCO-style annotations for each component
│   │       └── ...
│   ├── stage2_deformable_assembly/
│   │   ├── force_logs/
│   │   │   ├── 001_insertion_log.csv    # Logs of force/torque sensor data
│   │   │   └── ...
│   │   └── robotic_arm_poses/
│   │       ├── 001_assembly_poses.csv   # Trajectory data (joint angles, end-effector poses)
│   │       └── ...
│   └── stage3_final_assembly/
│       ├── correct_assembly_images/
│       │   ├── ideal_phone_1.png
│       │   └── ...
│       └── defect_images/
│           ├── misaligned_cpu.png       # For training the visual inspection AI
│           ├── missing_ram.png
│           ├── scratch_on_board.png
│           └── ...
```

---

## Outcomes and Metrices


### Key Outcomes

1.**Custom Multimodal Dataset**: A publicly available, high-quality dataset containing all the necessary URDFs, meshes, images, and sensor data to replicate the end-to-end assembly scenario.
2.**Integrated Multi-task Algorithm**: A working and fully documented algorithm that successfully performs object detection, force-guided assembly, AI-driven quality inspection, and logical decision-making.
3.**Reproducible Benchmark Suite**: A complete and self-contained ianvs project that allows other developers and researchers to easily run, test, and compare their own algorithms against our benchmark.


### Evaluation Metrics


The performance of any algorithm will be measured using the following key metrics:

| Metric Name | Description | Outcome/Value |
| :--- | :--- | :--- |
| **Assembly Accuracy** | Measures the success rate of the multi-stage assembly process. This is a binary Pass/Fail based on the visual inspection output. | `0` (Failure) or `1` (Success) |
| **Defect Detection Accuracy** | Measures the precision and recall of the CNN model in correctly identifying defects on the motherboard. | Percentage (`%`) |
| **Time Efficiency** | The total time taken for the robot to complete the entire workflow, from picking the first component to closing the box. | Time in seconds (`s`) |
| **Sorting Accuracy** | Measures the success of the robot's final decision—placing the correct USB cable in the box based on the component on the motherboard. | Percentage (`%`) |
| **Force Control Consistency** | Evaluates the smoothness and stability of the force-controlled assembly by analyzing sensor data logs. This ensures that delicate components are not damaged. | Quantitative score (e.g., Root Mean Square Error of force) |

---

## Road Map

The project is structured into three distinct phases, each with key deliverables that build on the previous month's work.


| Phase | Timeline | Key Deliverables |
| :--- | :--- | :--- |
| **Phase 1: Foundation & Data Generation** | September | **• Proposal Finalization:** Multiple iterations of proposal refinement.<br>**• Custom Dataset Creation:** Design and create all URDFs and `.obj` files for the robot and all components.<br>**• Dataset Generation Pipeline:** Develop the PyBullet script to generate the complete multi-modal dataset•  |
| **Phase 2: Core Algorithm & Integration** | October | **• Perception Module:** Implement the object detection algorithm (YOLOv8) to identify components on the pallet.<br>**• Manipulation Control:** Develop the force-controlled assembly logic for delicate component placement.<br>**• Quality Assurance Module:** Implement the CNN for visual inspection and defect detection.<br>**• `ianvs` Integration:** Integrate all algorithms and the custom dataset into the `ianvs` framework. |
| **Phase 3: Benchmarking & Reporting** | November | **• Decision-Making Logic:** Implement the `if-else` conditions for the quality check and the final cable sorting.<br>** • Data Publication:** Publish the custom dataset on Kaggle for public access.• End-to-End Workflow:** Fully integrate and test the entire multi-stage assembly and packaging pipeline.<br>**• Performance Benchmarking:** Execute the benchmark and collect data on all defined metrics.<br>**• Final Report & Documentation:** Generate the final report, leaderboard, and comprehensive documentation for the project. |

---

## Conclusion

The proposed embodied intelligence benchmarking framework for intelligent industrial assembly provides a comprehensive solution for evaluating multi-task robotic systems. By creating a custom, multi-modal dataset and an end-to-end assembly pipeline, this project addresses a critical gap in existing benchmarks. It moves beyond single-task scenarios and provides a real-world testbed that integrates:

1.**High-Precision Perception**: The use of algorithms like YOLOv8 and CNNs for both object detection and quality assurance.

2.**Intelligent Manipulation**: Force-controlled assembly that mimics human dexterity for delicate tasks.

3.**Logical Reasoning**: Conditional decision-making that allows the system to verify its own work and adapt to different scenarios.

This framework is designed as a complete example implementation that can be contributed to the ianvs repository, providing a valuable resource for the robotics and AI community. The impact of this project is significant, as it will serve as a foundation for developing the next generation of reliable, efficient, and truly intelligent autonomous systems for industrial manufacturing.

---