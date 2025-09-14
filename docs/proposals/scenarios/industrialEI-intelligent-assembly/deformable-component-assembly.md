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

2.**Develop an end-to-end assembly pipeline**: Implement a full assembly workflow that goes from component detection to placement to quality inspection.

3.**Integrate multi-model AI capabilities**: Combine object detection, RGB vision ,force-controlled manipulation that is the sense of touch and vision together with cohesive algorithm.

4.**Compile a list of key related research works**: Provide a list of relevant external datasets and algorithms (like REASSEMBLE ,DeepPCB and YOLOv8) that support the project's technical approach.

6.**Build a comprehensive benchmarking suite**: Develop a benchmarking suite within ianvs that includes standardized metrics, detailed test reports, and a leaderboard for evaluating multi-stage assembly performance.

---

## Scope

1.**Create the Dataset**: Build a new, multimodal dataset that includes both visual and touch sensor data.

2.**Develop Simple Algorithm Modules**: Create two simple, sequential modules: an object detection module (to find the component) and an assembly module (to place it). These algorithms will demonstrate that the new dataset is usable.

3.**Define End-to-End Metrics**: Instead of focusing on metrics for each individual sub-task (like YOLO accuracy), define one or a few end-to-end metrics to evaluate the complete assembly process, such as the overall success rate or final assembly accuracy. This aligns with the "multi-stage" nature of the project.

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

 This  project addresses this by using an embodied intelligent robotic arm. This robotic arm leverages a fusion of advanced capabilities, including **force detection** for delicate haptic control, **visual detection** for accurate positioning. By integrating these systems, the robot can not only assemble components with superior precision but also perform real-time quality checks to ensure a perfect final product.
[Image description](images/working_assembly_workflow.png)

---

## Ianvs 

### Ianvs Framework Integration Architecture

The deformable component algorithm is designed to integrate seamlessly with the Ianvs framework through its three core components, as illustrated in the following architecture diagram:

![Ianvs Framework Architecture](images\benchmarking_architecture.png)

*Figure above: Ianvs framework architecture showing the integration of Test Environment Manager, Test Case Controller, and Story Manager for benchmarking embodied intelligence in industrial environments.*

The initial focus is on configuring the custom, multi-modal `Intelligent Assembly Dataset` within the Test Environment Manager and establishing the multi-stage performance metrics. Simultaneously, the `Test Case Controller` will be implemented with a new Multi-stage Assembly Workflow. This paradigm will perform comprehensive benchmark tests on the custom dataset, evaluating the full, end-to-end workflow—from component positioning with `YOLOv8`.


### Test Environment Manager Integration

This project will be configured and deployed by defining a new `Test Environment` that specifies all the necessary components for the benchmarking process.

1.**Dataset Configuration**: The ianvs environment will use the custom dataset hosted on Kaggle, including URDFs, images, and sensor data logs.
2.**Sensor Configuration**: Parameters for the simulated camera (RGB-D) and the force/torque sensor will be defined.
3.**Algorithm Parameters**: The configuration will include hyperparameters for our core algorithms, like YOLOv8 and force control logic.
4.**System Constraints**: Define hardware and software constraints, such as processing time and memory limits, to ensure reproducibility.

### Test Case Controller Implementation

The `Test Case Controller` will be responsible for orchestrating the execution of our end-to-end assembly workflow.

1.**Algorithm Template**: Implement a template that defines the sequential, multi-stage process: perception → manipulation → verification.
2.**Data Loading**: Automated downloading and loading of our custom dataset from Kaggle.
3.**Execution Pipeline**: The controller will execute the full workflow step-by-step, running object detection, assembly, and visual inspection.

### Story Manager Output

The `Story Manager` will handle the output and presentation of our benchmark results, providing clear insights into the algorithm's performance.

1.**Metrics Calculation**: Compute our primary end-to-end metric: Assembly Success Rate.
2.**Leaderboard Generation**: The results will be used to generate a leaderboard that ranks algorithm performance.
3.**Report Generation**: A detailed report with a full analysis of the benchmark results will be produced.


###  Directory Structure

The new dataset will be generated and uploaded in the https://www.kaggle.com/datasets/kubeedgeianvs ,users can install the zip file and then unzip in in their datasets folder .

```
──ianvs/examples/industrialEI/
            deformable__assembly/
                  ├── testalgorithms/
                  │   └── assembly_alg/   
                  │       ├── __init__.py
                  │       ├── basemodel.py            # Core pose estimation algorithm
                  │       ├── data_utils.py           
                  │       ├── eval.py
                  │       ├── robot_control.py
                  │       ├── perception.py           # YOLO model for object detection and CNN for visual inspection
                  │       └── assembly_planner.py     #  Orchestrates the entire multi-step component assembly process
                  ├── testenv/
                  │   └── assembly__env/    
                  │       ├── __init__.py
                  │       ├── testenv.yaml
                  |       ├── acc.py                  # Evaluation metrics
                  ├── .gitignore
                  ├── LICENSE
                  ├── README.md                       # documentation for users
                  └── benchmarkingjob.yaml
```


### **Multi-stage Assembly Workflow**

This proposal focuses on a multi-stage, single-task assembly process, which is a more accurate representation of real-world industrial workflows. The approach integrates multiple, sequential AI capabilities into a single cohesive pipeline. It demonstrates an embodied intelligent system that can perceive, manipulate, verify, and reason in a step-by-step manner.

The Single-Task, Multi-stage Paradigm works as follows:

1.A developer deploys an application that orchestrates the entire assembly process.
2.The application launches the full workflow, which executes a series of interdependent, sequential tasks.
3.The robotic arm uses **YOLOv8** to perform real-time object detection on components from the panel.
4.The robot then executes a force-controlled assembly task using its touch sensors to place the components on the motherboard precisely.
5.The system then generates a final test report and updates the leaderboard, concluding the process.

[Image description](images/benchmarking_architecture.png)

---

## Algorithm

### **Algorithm**

| Target/Object | Input Data | Common Industrial Algorithms |
| :--- | :--- | :--- |
| **Component Positioning & Grasping** | Live RGB-D camera feed `(PyBullet)` | `YOLOv8` (for object detection) |
| **Robotic Arm Trajectory & Grasping** | Object position ($x, y, z$), orientation, component URDFs | GraspNet, RRT* (for path planning), Kinematics/Inverse Kinematics |
| **Deformable Assembly** | Live force/torque sensor data, joint angles | Force/Impedance Control |
| **Visual Inspection & Verification** | RGB image of assembled motherboard | `Convolutional Neural Network (CNN)` for Image Classification or Anomaly Detection |

---

## Dataset Management

Different datasets will be used to train the robotic AI model:

1.**The New Dataset** : It is the custom-made data from PyBullet simulation, including the URDFs, images of the assembled phone, and sensor logs.

2.**REASSEMBLE** :  Used as a "practice test" to train and validate the fundamental skills of your robotic arm, especially for force-controlled insertion.

3.**DeepPCB** : It's a collection of images of electronic circuit boards with defects. This data is used to train quality control AI to recognize defects on a motherboard.


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
2.**Integrated Multi-task Algorithm**: A working and fully documented algorithm that serves as a reproducible baseline for the multi-stage workflow, successfully performing object detection and force-guided assembly.
3.**Reproducible Benchmark Suite**: A complete and self-contained `ianvs` project that allows other developers and researchers to easily run, test, and compare their own algorithms against our benchmark.


### Evaluation Metrics

The performance of any algorithm will be measured by a suite of end-to-end metrics that reflect real-world performance. These metrics will evaluate not only the final outcome but also the efficiency and precision of the entire multi-stage process.


| Metric Name | Description | Outcome/Value |
| :--- | :--- | :--- |
| **Assembly Accuracy** | Measures the success rate of the multi-stage assembly process. This is a binary Pass/Fail based on the visual inspection output. | `0` (Failure) or `1` (Success) |

---

## Future Work and Stretch Goals

To maintain a clear and focused scope for this initial project, certain advanced features will be considered as future work. These enhancements, once implemented, will further validate the complexity and richness of our custom dataset.

1.**Intelligent Decision-Making (Cable Sorting Logic)**: After a successful assembly and quality check, the robot could implement a secondary logical task. This would involve a conditional if-else check to determine which accessory (e.g., USB-C or USB-B cable) to include with the final product. This would demonstrate the robot's ability to reason and adapt its actions based on specific components used during assembly.

2.**Complex Logical Control Flows**: Likely explore implementing more complex logical control flows beyond simple `if-else conditions. This includes advanced state machines and adaptive reasoning that would allow the robot to handle unexpected events, such as re-attempting a component placement if the initial visual inspection fails. These additions would push the boundaries of current embodied intelligence benchmarks.

---

## Road Map

The project is structured into three distinct phases, each with key deliverables that build on the previous month's work.


| Phase | Timeline | Key Deliverables |
| :--- | :--- | :--- |
| **Phase 1: Foundation & Data Generation** | September | **• Proposal Finalization:** Multiple iterations of proposal refinement.<br>**• Custom Dataset Creation:** Design and create all URDFs and `.obj` files for the robot and all components.<br>**• Dataset Generation Pipeline:** Develop the PyBullet script to generate the complete multi-modal dataset•  |
| **Phase 2: Core Algorithm & Integration** | October | **• Perception Module:** Implement the object detection algorithm (YOLOv8) to identify components on the pallet.<br>**• Manipulation Control:** Develop the force-controlled assembly logic for delicate component placement.<br>**• Quality Assurance Module:** Implement the CNN for visual inspection and defect detection.<br>**• `ianvs` Integration:** Integrate all algorithms and the custom dataset into the `ianvs` framework. |
| **Phase 3: Benchmarking & Reporting** | November | **• Decision-Making Logic:** Implement the `if-else` conditions for the quality check and the final cable sorting.<br>** • Data Publication:** Publish the custom dataset on Kaggle for public access.• **End-to-End Workflow:** Fully integrate and test the entire multi-stage assembly and packaging pipeline.<br>**• Performance Benchmarking:** Execute the benchmark and collect data on all defined metrics.<br>**• Final Report & Documentation:** Generate the final report, leaderboard, and comprehensive documentation for the project. |

---

## Conclusion

The proposed embodied intelligence benchmarking framework for intelligent industrial assembly provides a comprehensive solution for evaluating robotic systems. By creating a custom, multi-modal dataset and an end-to-end assembly pipeline, this project addresses a critical gap in existing benchmarks. It provides a real-world testbed for a single, complex task that integrates:

1.**High-Precision Perception**: The use of algorithms like YOLOv8 and CNNs for both object detection and quality assurance.

2.**Intelligent Manipulation**: Force-controlled assembly that mimics human dexterity for delicate tasks.

3.End-to-End Evaluation: A complete pipeline that allows the system to verify its own work and provides objective performance metrics.

This framework is designed as a complete example implementation that can be contributed to the ianvs repository, providing a valuable resource for the robotics and AI community. The impact of this project is significant, as it will serve as a foundation for developing the next generation of reliable and efficient autonomous systems for industrial manufacturing.

---