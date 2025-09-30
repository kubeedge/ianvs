# Physically Consistent and Interactable Indoor Simulation Scene Generation: Implementation Based on KubeEdge-Ianvs

## About the Project

This project is an open-source framework designed to automatically generate and evaluate physically realistic 3D indoor scenes from textual descriptions, based on Holodeck [^1] and PhyScene [^2]. The implementation leverages the official codebases for [Holodeck](https://github.com/allenai/Holodeck) and [PhyScene](https://github.com/PhyScene/PhyScene), adapting them for our specific framework.

This project provides a standardized platform for researchers and developers to easily test, evaluate, and compare different Text-to-Scene generation algorithms. It includes an end-to-end pipeline, featuring a sample dataset, a baseline generation algorithm, and a suite of evaluation metrics to assess the quality and semantic accuracy of the generated scenes.

**Key Features**

- End-to-End Benchmarking: A complete toolchain from text prompt input to scene generation and quantitative evaluation.

- Multi-dimensional Evaluation: Assesses not only the physical plausibility of scenes but also their semantic conformance to the input text.

- Standardized Dataset: Includes a set of standard input queries to ensure a fair comparison between different algorithms.

## Getting Started

### Environment Setup


#### 1. Create conda environment

It is strongly recommended to use conda to isolate the environment.

```bash
conda create --name phys_scene_gen python=3.10.18
conda activate phys_scene_gen
```

#### 2. Install the ianvs framework

```bash
# Confirm that you are already in the ianvs root directory
pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl
pip install -r requirements.txt
pip install -e .
```

#### 3. Install dependencies for this benchmark

Navigate to the root directory of this project (phys_scene_gen/singletask_learning_bench) and install the additional dependencies required for the Holodeck algorithm. The `numpy` library is specified with a version at the end to avoid conflicts in this code base.

```bash
cd examples/phys_scene_gen/singletask_learning_bench
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+3213d486cd09bcbafce33561997355983bdf8d1a
pip install numpy==1.24.3
```

#### 4. Prepare Holodeck data and assets

Running Holodeck requires pre-downloading 3D model assets and associated feature data. Please follow the instructions in the [official Holodeck documentation](https://github.com/allenai/Holodeck/blob/main/README.md) to download the required data. Here is an example:

```bash
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23
python -m objathor.dataset.download_assets --version 2023_09_23
python -m objathor.dataset.download_annotations --version 2023_09_23
python -m objathor.dataset.download_features --version 2023_09_23
```

By default, the data will be downloaded to the `~/.objathor-assets/` directory.

### Configure the Benchmark

Before running, please check and modify the following key configurations in the `testalgorithms/ai2holodeck/test_holodeck.yaml` file:

```yaml
      hyperparameters:
        - openai_api_key:
            values:
              - "sk-YOUR_OPENAI_API_KEY_HERE"
        - objaverse_asset_dir:
            values:
              - "YOUR_ROOT_TO_OBJAVERSE_ASSET"  # default: "~/.objathor-assets/"
        - single_room:
            values:
              - False
        - data_path:
            values:
              - "./examples/phys_scene_gen/singletask_learning_bench/dataset/queries.jsonl"
```

### Create the Input File

Navigate to the root directory of this project (phys_scene_gen/singletask_learning_bench). Create a `dataset` (phys_scene_gen/singletask_learning_bench/dataset) folder and create a `queries.jsonl` file in it. Then write your queries after `"answer"` and leave `"question"` blank. Here is an example:

```json
{"question": "", "answer": "a modern living room"}
{"question": "", "answer": "a cozy bedroom with a wooden bed"}
```

## Run the Benchmark

Once all configurations are set, confirm project folder structure, navigate to the ianvs root directory and use the ianvs command-line tool to start the benchmark job. Run:

```bash
# Confirm that you are already in the ianvs root directory and have activated conda environment
ianvs -f examples/phys_scene_gen/singletask_learning_bench/benchmarkingjob.yaml
```

All generated files and the final evaluation report in the `examples/phys_scene_gen/singletask_learning_bench/workspace/` directory, and the generated scenes in the `generated_scenes/` directory.

## Project Folder Structure

The initial project folder structure should look like this:

```
ianvs/
└── examples/
    └── phys_scene_gen/
        └── singletask_learning_bench/
            ├── testalgorithms/
            │   └── ai2holodeck/
            │       ├── generation/
            │       ├── __init__.py
            │       ├── constants.py
            │       ├── main.py
            │       ├── test_holodeck.yaml
            │       └── two_stage_generator.py
            │
            ├── testenv/
            │   ├── metrics/
            │   │   ├── __init__.py
            │   │   └── semantic_conformance.py
            │   └── testenv.yaml
            │
            ├── benchmarkingjob.yaml
            ├── readme.md
            └── requirements.txt
```

## References

[^1]: Yang, Y., Sun, F.-Y., Weihs, L., VanderBilt, E., Herrasti, A., Han, W., Wu, J., Haber, N., Krishna, R., Liu, L., Callison-Burch, C., Yatskar, M., Kembhavi, A., & Clark, C. (2024). Holodeck: Language Guided Generation of 3D Embodied AI Environments. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16227–16237. Code available at: [https://github.com/allenai/Holodeck](https://github.com/allenai/Holodeck)

[^2]: Yang, Y., Jia, B., Zhi, P., & Huang, S. (2024). PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI. Proceedings of Conference on Computer Vision and Pattern Recognition (CVPR). Code available at: [https://github.com/PhyScene/PhyScene](https://github.com/PhyScene/PhyScene)