# Ianvs v0.1.0 release
## 1. Release the Ianvs distributed synergy AI benchmarking framework.
   a) Release test environment management and configuration.
   b) Release test case management and configuration.
   c) Release test story management and configuration.
   d) Release the open-source test case generation tool: Use hyperparameter enumeration to fill in one configuration file to generate multiple test cases.

## 2. Release the PCB-AoI public dataset.
Release the PCB-AoI public dataset, its corresponding preprocessing, and baseline algorithm projects. 
Ianvs is the first open-source site for that dataset.

## 3. Support two new paradigms in test environments and test cases. 
   a) Test environments and test cases that support the single-task learning paradigm.
   b) Test environments and test cases that support the incremental learning paradigm.

## 4. Release PCB-AoI benchmark cases based on the two new paradigms.
   a) Release PCB-AoI benchmark cases based on single-task learning, including leaderboards and test reports.
   b) Release PCB-AoI benchmark cases based on incremental learning, including leaderboards and test reports.

# Ianvs v0.2.0 release

This version of Ianvs supports the following functions of unstructured lifelong learning:

## 1. Support lifelong learning throughout the entire lifecycle, including task definition, task assignment, unknown task recognition, and unknown task handling, among other modules, with each module being decoupled.
   - Support unknown task recognition and provide corresponding usage examples based on semantic segmentation tasks in [this example](https://github.com/kubeedge/ianvs/tree/main/examples/robot-cityscapes-synthia/lifelong_learning_bench/semantic-segmentation).
   - Support multi-task joint inference and provide corresponding usage examples based on object detection tasks in [this example](https://github.com/kubeedge/ianvs/tree/main/examples/MOT17/multiedge_inference_bench/pedestrian_tracking).

## 2. Provide classic lifelong learning testing metrics, and support for visualizing test results.
   - Support lifelong learning system metrics such as BWT and FWT.
   - Support visualization of lifelong learning results.
   
## 3. Provide real-world datasets and rich examples for lifelong learning testing, to better evaluate the effectiveness of lifelong learning algorithms in real environments.
   - Provide cloud-robotics datasets in [this website](https://kubeedge-ianvs.github.io/).
   - Provide cloud-robotics semantic segmentation examples in [this example](https://github.com/kubeedge/ianvs/tree/main/examples/robot/lifelong_learning_bench).