# Testing Open-set recognition in Curb-detection datasets

## About Open-set recognition

Traditional classiﬁers are deployed under a closed-set setting, with both training and test classes belonging to the same set. However, real-world applications probably face the input of unknown categories, and the model will recognize them as known ones. Under such circumstances, open-set recognition is proposed to maintain classiﬁcation performance on known classes and reject unknowns. The closed-set models make overconﬁdent predictions over familiar known class instances so that calibration and thresholding across categories become essential issues when extending to an open-set environment. 

This test aims to reproduce the CVPR2021 paper "Learning placeholders for open-set recognition".

See [here](https://github.com/Frank-lilinjie/ianvs/tree/feature-lifelong-n/docs/proposals/algorithms/lifelong-learning/Unknown-task-recognition/Open-set recognition Learning Placeholders.md) for details



## About Curb-detection datasets

Two datasets, SYNTHIA, and cityscape, were selected for this project. Because SYNTHIA is often easier to obtain than the real urban road dataset as simulated by the simulator in a real research environment, it is treated as known task data for model pre-training, while the real urban landscape image acquisition requires more resources and is more difficult to obtain, so it is treated as unknown task data.

See [here](https://github.com/Frank-lilinjie/ianvs/tree/feature-lifelong-n/docs/proposals/algorithms/lifelong-learning/Unknown-task-recognition/curb_detetion_datasets.md) for details



## Benchmark Setting

Key settings of the test environment to Open-set recognition are as follows:

```yaml
benchmarkingjob:
  # job name of bechmarking; string type;
  name: "benchmarkingjob"
  # the url address of job workspace that will reserve the output of tests; string type;
  workspace: "/ianvs/lifelong_learning_bench/workspace"

  # the url address of test environment configuration file; string type;
  # the file format supports yaml/yml;
  testenv: "./examples/curb-detection/lifelong_learning_bench/testenv/testenv.yaml"

  # the configuration of test object
  test_object:
    # test type; string type;
    # currently the option of value is "algorithms",the others will be added in succession.
    type: "algorithms"
    # test algorithm configuration files; list type;
    algorithms:
      # algorithm name; string type;
      - name: "rfnet_lifelong_learning"
        # the url address of test algorithm configuration file; string type;
        # the file format supports yaml/yml
        url: "./examples/curb-detection/lifelong_learning_bench/testalgorithms/rfnet/rfnet_algorithm.yaml"
```

Key settings of the algorithm to incremental learning are as follows:

```yaml
algorithm:
  # paradigm type; string type;
  # currently the options of value are as follows:
  #   1> "singletasklearning"
  #   2> "incrementallearning"
  #   3> "lifelonglearning"
  paradigm_type: "lifelonglearning"
  lifelong_learning_data_setting:
    # ratio of training dataset; float type;
    # the default value is 0.8.
    train_ratio: 0.8
    # the method of splitting dataset; string type; optional;
    # currently the options of value are as follows:
    #   1> "default": the dataset is evenly divided based train_ratio;
    splitting_method: "default"

  # algorithm module configuration in the paradigm; list type;
  modules:
    # type of algorithm module; string type;
    # currently the options of value are as follows:
    #   1> "basemodel": contains important interfaces such as train、 eval、 predict and more; required module;
    - type: "basemodel"
      # name of python module; string type;
      # example: basemodel.py has BaseModel module that the alias is "FPN" for this benchmarking;
      name: "BaseModel"
      # the url address of python module; string type;
      url: "./examples/curb-detection/lifelong_learning_bench/testalgorithms/rfnet/basemodel.py"
      # hyperparameters configuration for the python module; list type;
      hyperparameters:
        # name of the hyperparameter; string type;
        - learning_rate:
            values:
              - 0.0001
    #  2> "task_definition": define lifelong task ; optional module;
    - type: "task_definition"
      # name of python module; string type;
      name: "TaskDefinitionByOrigin"
      # the url address of python module; string type;
      url: "./examples/curb-detection/lifelong_learning_bench/testalgorithms/rfnet/task_definition_by_origin.py"
      # hyperparameters configuration for the python module; list type;
      hyperparameters:
        # name of the hyperparameter; string type;
        # origins of data; value is ["real", "sim"], this means that data from real camera and simulator.
        - origins:
            values:
              - [ "real", "sim" ]
    #  3> "task_allocation": allocate lifelong task ; optional module;
    - type: "task_allocation"
      # name of python module; string type;
      name: "TaskAllocationByOrigin"
      # the url address of python module; string type;
      url: "./examples/curb-detection/lifelong_learning_bench/testalgorithms/rfnet/task_allocation_by_origin.py"
      # hyperparameters configuration for the python module; list type;
      hyperparameters:
        # name of the hyperparameter; string type;
        # origins of data; value is ["real", "sim"], this means that data from real camera and simulator.
        - origins:
            values:
              - [ "real", "sim" ]
    - type: "unknow_task_recognition"
      # name of python module; string type;
      name: "SampleRegonitionByScene"
      # the url address of python module; string type;
      url: "./examples/curb-detection/lifelong_learning_bench/testalgorithms/rfnet/unknow_task_recognition.py"
       # hyperparameters configuration for the python module; list type;
      hyperparameters:
        -model_path:
          values:
            - "./models/test_model_scene/Epochofprose7.pth"
        -path_inference_sample:
          values:
            - "./datasets/inference_dataset/inference/inference"
        -path_inference_dataset:
          values:
            - "./datasets/inference_dataset/inference"
        -path_seen_sample:
          values:
            - "./datasets/inference_dataset/seen_sample"
        -path_unseen_sample:
          values:
            - "./datasets/inference_dataset/unseen_sample"
```



## Benchmark Result



## Effect Display