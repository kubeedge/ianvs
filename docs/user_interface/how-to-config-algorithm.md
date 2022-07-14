# How to config algorithm

The algorithm developer is able to test his/her own targeted algorithm and configs the algorithm using the following
configuration.

## The configuration of algorithm

| Property | Required | Description |
|----------|----------|-------------|
|paradigm_type|yes|Paradigm name; Type: string; Value Constraint: Currently the options of value are as follows: 1> singletasklearning 2> incrementallearning|
|incremental_learning_data_setting|no|Data setting for incremental learning paradigm.[the configuration of incremental_learning_data_setting](#id1)|
|initial_model_url|no|The url address of initial model for model pre-training; Type: string|
|modules|yes|The algorithm modules for paradigm; Type: list; Value Constraint: the list of [the configuration of module](#id2)|

For example:

```yaml
algorithm:
  # paradigm type; string type;
  # currently the options of value are as follows:
  #   1> "singletasklearning"
  #   2> "incrementallearning"
  paradigm_type: "incrementallearning"
  incremental_learning_data_setting:
    ...
  # the url address of initial model for model pre-training; string url;
  initial_model_url: "/ianvs/initial_model/model.zip"

  # algorithm module configuration in the paradigm; list type;
  modules:
    ...
```

### The configuration of incremental_learning_data_setting

| Property | Required | Description |
|----------|----------|-------------|
|train_ratio|no|Ratio of training dataset; Type: float; Default value: 0.8; Value Constraint: the value is greater than 0 and less than 1.|
|splitting_method|no|The method of splitting dataset; Type: string; Default value: default; Value Constraint: Currently the options of value are as follows: 1> default: the dataset is evenly divided based train_ratio.

For example:

```yaml
incremental_learning_data_setting:
  # ratio of training dataset; float type;
  # the default value is 0.8.
  train_ratio: 0.8
  # the method of splitting dataset; string type; optional;
  # currently the options of value are as follows:
  #   1> "default": the dataset is evenly divided based train_ratio;
  splitting_method: "default"
```

### The configuration of module

| Property | Required | Description |
|----------|----------|-------------|
|type|yes|Algorithm module type; Type: string; Value Constraint: Currently the options of value are as follows: 1> basemodel: the algorithm module contains important interfaces such as train, eval, predict and more.it's required module. 2> hard_example_mining: the module checks hard example when predict. it's optional module and often used for incremental learning paradigm. |
|name|yes|Algorithm module name; Type: string; Value Constraint: a python module name|
|url|yes|The url address of python module file; Type: string |
|hyperparameters|no|[the configuration of hyperparameters](#id3)|

For example:

```yaml
# algorithm module configuration in the paradigm; list type;
modules:
  # type of algorithm module; string type;
  # currently the options of value are as follows:
  #   1> "basemodel": contains important interfaces such as train、 eval、 predict and more; required module;
  - type: "basemodel"
    # name of python module; string type;
    # example: basemodel.py has BaseModel module that the alias is "FPN" for this benchmarking;
    name: "FPN"
    # the url address of python module; string type;
    url: "./examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn/basemodel.py"

    # hyperparameters configuration for the python module; list type;
    hyperparameters:
      ...
    #  2> "hard_example_mining": check hard example when predict ; optional module;
  - type: "hard_example_mining"
    # name of python module; string type;
    name: "IBT"
    # the url address of python module; string type;
    url: "./examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn/hard_example_mining.py"
    # hyperparameters configuration for the python module; list type;
    hyperparameters:
      ...
```

### The configuration of hyperparameters

The following is an example of hyperparameters configuration:

```yaml
# hyperparameters configuration for the python module; list type;
hyperparameters:
  # name of the hyperparameter; string type;
  - momentum:
    # values of the hyperparameter; list type;
    # types of the value are string/int/float/boolean/list/dictionary
      values:
        - 0.95
        - 0.5
  - learning_rate:
      values:
        - 0.1
        - 0.2
```

Ianvs will test for all the hyperparameter combination, that means it will run all the following 4 test:

| Num  | learning_rate | momentum |
|------|---------------|----------|
| 1    | 0.1           | 0.95     |
| 2    | 0.1           | 0.5      |
| 3    | 0.2           | 0.95     |
| 4    | 0.2           | 0.5      |

Currently, Ianvs is not restricted to validity of the hyperparameter combination. That might lead to some invalid
parameter combination, and it is controlled by the user himself. In the further version of Ianvs, it will support
excluding invalid parameter combinations to improve efficiency.

## Show example

```yaml
# fpn_algorithm.yaml
algorithm:
  # paradigm type; string type;
  # currently the options of value are as follows:
  #   1> "singletasklearning"
  #   2> "incrementallearning"
  paradigm_type: "incrementallearning"
  incremental_learning_data_setting:
    # ratio of training dataset; float type;
    # the default value is 0.8.
    train_ratio: 0.8
    # the method of splitting dataset; string type; optional;
    # currently the options of value are as follows:
    #   1> "default": the dataset is evenly divided based train_ratio;
    splitting_method: "default"
  # the url address of initial model for model pre-training; string url;
  initial_model_url: "/ianvs/initial_model/model.zip"

  # algorithm module configuration in the paradigm; list type;
  modules:
    # type of algorithm module; string type;
    # currently the options of value are as follows:
    #   1> "basemodel": contains important interfaces such as train、 eval、 predict and more; required module;
    - type: "basemodel"
      # name of python module; string type;
      # example: basemodel.py has BaseModel module that the alias is "FPN" for this benchmarking;
      name: "FPN"
      # the url address of python module; string type;
      url: "./examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn/basemodel.py"

      # hyperparameters configuration for the python module; list type;
      hyperparameters:
        # name of the hyperparameter; string type;
        - momentum:
            # values of the hyperparameter; list type;
            # types of the value are string/int/float/boolean/list/dictionary
            values:
              - 0.95
              - 0.5
        - learning_rate:
            values:
              - 0.1
      #  2> "hard_example_mining": check hard example when predict ; optional module;
    - type: "hard_example_mining"
      # name of python module; string type;
      name: "IBT"
      # the url address of python module; string type;
      url: "./examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn/hard_example_mining.py"
      # hyperparameters configuration for the python module; list type;
      hyperparameters:
        # name of the hyperparameter; string type;
        # threshold of image; value is [0, 1]
        - threshold_img:
            values:
              - 0.9
        # predict box of image; value is [0, 1]
        - threshold_box:
            values:
              - 0.9
```