# How to config benchmarkingjob

The algorithm developer is able to test his/her own targeted algorithm using the following configuration information.

## The configuration of benchmarkingjob

| Property | Required | Description |
|----------|----------|-------------|
|name|yes|Job name of benchmarking; Type: string|
|workspace|no|The url address of job workspace that will reserve the output of tests; Type: string; Default value: `./workspace`|
|testenv|yes|The url address of test environment configuration file; Type: string; Value Constraint: The file format supports yaml/yml.|
|test_object|yes|[The configuration of test_object](#id1)|
|rank|yes|[The configuration of ranking leaderboard](#id2)|

For example:

```yaml
benchmarkingjob:
  # job name of benchmarking; string type;
  name: "benchmarkingjob"
  # the url address of job workspace that will reserve the output of tests; string type;
  # default value: "./workspace"
  workspace: "/ianvs/incremental_learning_bench/workspace"

  # the url address of test environment configuration file; string type;
  # the file format supports yaml/yml;
  testenv: "./examples/pcb-aoi/incremental_learning_bench/testenv/testenv.yaml"
  # the configuration of test object
  test_object:
    ...
  # the configuration of ranking leaderboard
  rank:
    ...
```

### The configuration of test_object

| Property | Required | Description |
|----------|----------|-------------|
|type|yes|Type of test object; Type: string; Value Constraint: Currently the option of value is "algorithms",the others will be added in succession.|
|algorithms|no|[Test algorithm configuration](#id2); Type: list|

For example:

```yaml
# the configuration of test object
test_object:
  # test type; string type;
  # currently the option of value is "algorithms",the others will be added in succession.
  type: "algorithms"
  # test algorithm configuration files; list type;
  algorithms:
    ...
```

### The configuration of algorithms

| Property | Required | Description |
|----------|----------|-------------|
|name|yes|Algorithm name; Type: string|
|url|yes|The url address of test algorithm configuration file; Type: string; Value Constraint: The file format supports yaml/yml.|

For example:

```yaml
# test algorithm configuration files; list type;
algorithms:
  # algorithm name; string type;
  - name: "fpn_incremental_learning"
  # the url address of test algorithm configuration file; string type;
  # the file format supports yaml/yml
  url: "./examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn/fpn_algorithm.yaml"
```

### The configuration of rank

| Property | Required | Description |
|----------|----------|-------------|
|sort_by|yes|Rank leaderboard with metric of test case's evaluation and order; Type: list; Value Constraint: The sorting priority is based on the sequence of metrics in the list from front to back.|
|visualization|yes|[The configuration of visualization](#id3)|
|selected_dataitem|yes|[The configuration of selected_dataitem](#id4); The user can add his/her interested dataitems in terms of "paradigms", "modules", "hyperparameters" and "metrics", so that the selected columns will be shown.|
|save_mode|yes|save mode of selected and all dataitems in workspace `./rank`; Type: string; Value Constraint: Currently the options of value are as follows: 1> "selected_and_all": save selected and all dataitems. 2> "selected_only": save selected dataitems.|

For example:

```yaml
# the configuration of ranking leaderboard
rank:
  # rank leaderboard with metric of test case's evaluation and order ; list type;
  # the sorting priority is based on the sequence of metrics in the list from front to back;
  sort_by: [ { "f1_score": "descend" }, { "samples_transfer_ratio": "ascend" } ]
  # visualization configuration
  visualization:
    ...
  # selected dataitem configuration
  # The user can add his/her interested dataitems in terms of "paradigms", "modules", "hyperparameters" and "metrics",
  # so that the selected columns will be shown.
  selected_dataitem:
    ...
  # save mode of selected and all dataitems in workspace `./rank` ; string type;
  # currently the options of value are as follows:
  #  1> "selected_and_all": save selected and all dataitems;
  #  2> "selected_only": save selected dataitems;
  save_mode: "selected_and_all"
```

### The configuration of visualization

| Property | Required | Description |
|----------|----------|-------------|
|mode|no|Mode of visualization in the leaderboard. There are quite a few possible dataitems in the leaderboard. Not all of them can be shown simultaneously on the screen; Type: string; Default value: selected_only|
|method|no|Method of visualization for selected dataitems; Type: string; Value Constraint: Currently the options of value are as follows: 1> "print_table": print selected dataitems.|

For example:

```yaml
# visualization configuration
visualization:
# mode of visualization in the leaderboard; string type;
# There are quite a few possible dataitems in the leaderboard. Not all of them can be shown simultaneously on the screen.
# In the leaderboard, we provide the "selected_only" mode for the user to configure what is shown or is not shown.
mode: "selected_only"
# method of visualization for selected dataitems; string type;
# currently the options of value are as follows:
#  1> "print_table": print selected dataitems;
method: "print_table"
```

### The configuration of selected_dataitem

| Property | Required | Description |
|----------|----------|-------------|
|paradigms|yes|Select paradigms in the leaderboard; Type: list; Default value: ["all"]; Value Constraint: Currently the options of value are as follows: 1> "all": select all paradigms in the leaderboard. 2> paradigms in the leaderboard, e.g., "singletasklearning".|
|modules|yes|Select modules in the leaderboard; Type: list; Default value: ["all"]; Value Constraint: Currently the options of value are as follows: 1> "all": select all hyperparameters in the leaderboard. 2> hyperparameters in the leaderboard, e.g., "momentum".|
|hyperparameters|yes|Select hyperparameters in the leaderboard; Type: list; Default value: ["all"]; Value Constraint: Currently the options of value are as follows: 1> "all": select all hyperparameters in the leaderboard. 2> hyperparameters in the leaderboard, e.g., "momentum".|
|metrics|yes|Select metrics in the leaderboard; Type: list; Default value: ["all"]; Value Constraint: Currently the options of value are as follows: 1> "all": select all metrics in the leaderboard. 2> metrics in the leaderboard, e.g., "f1_score".|

```yaml
# selected dataitem configuration
# The user can add his/her interested dataitems in terms of "paradigms", "modules", "hyperparameters" and "metrics",
# so that the selected columns will be shown.
selected_dataitem:
  # currently the options of value are as follows:
  #   1> "all": select all paradigms in the leaderboard;
  #   2> paradigms in the leaderboard, e.g., "singletasklearning"
  paradigms: [ "all" ]
  # currently the options of value are as follows:
  #   1> "all": select all modules in the leaderboard;
  #   2> modules in the leaderboard, e.g., "basemodel"
  modules: [ "all" ]
  # currently the options of value are as follows:
  #   1> "all": select all hyperparameters in the leaderboard;
  #   2> hyperparameters in the leaderboard, e.g., "momentum"
  hyperparameters: [ "all" ]
  # currently the options of value are as follows:
  #   1> "all": select all metrics in the leaderboard;
  #   2> metrics in the leaderboard, e.g., "F1_SCORE"
  metrics: [ "f1_score", "samples_transfer_ratio" ]
```

## Show the example

```yaml
benchmarkingjob:
  # job name of benchmarking; string type;
  name: "benchmarkingjob"
  # the url address of job workspace that will reserve the output of tests; string type;
  # default value: "./workspace"
  workspace: "/ianvs/incremental_learning_bench/workspace"

  # the url address of test environment configuration file; string type;
  # the file format supports yaml/yml;
  testenv: "./examples/pcb-aoi/incremental_learning_bench/testenv/testenv.yaml"

  # the configuration of test object
  test_object:
    # test type; string type;
    # currently the option of value is "algorithms",the others will be added in succession.
    type: "algorithms"
    # test algorithm configuration files; list type;
    algorithms:
      # algorithm name; string type;
      - name: "fpn_incremental_learning"
        # the url address of test algorithm configuration file; string type;
        # the file format supports yaml/yml
        url: "./examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn/fpn_algorithm.yaml"

  # the configuration of ranking leaderboard
  rank:
    # rank leaderboard with metric of test case's evaluation and order ; list type;
    # the sorting priority is based on the sequence of metrics in the list from front to back;
    sort_by: [ { "f1_score": "descend" }, { "samples_transfer_ratio": "ascend" } ]

    # visualization configuration
    visualization:
      # mode of visualization in the leaderboard; string type;
      # There are quite a few possible dataitems in the leaderboard. Not all of them can be shown simultaneously on the screen.
      # In the leaderboard, we provide the "selected_only" mode for the user to configure what is shown or is not shown.
      mode: "selected_only"
      # method of visualization for selected dataitems; string type;
      # currently the options of value are as follows:
      #  1> "print_table": print selected dataitems;
      method: "print_table"

    # selected dataitem configuration
    # The user can add his/her interested dataitems in terms of "paradigms", "modules", "hyperparameters" and "metrics",
    # so that the selected columns will be shown.
    selected_dataitem:
      # currently the options of value are as follows:
      #   1> "all": select all paradigms in the leaderboard;
      #   2> paradigms in the leaderboard, e.g., "singletasklearning"
      paradigms: [ "all" ]
      # currently the options of value are as follows:
      #   1> "all": select all modules in the leaderboard;
      #   2> modules in the leaderboard, e.g., "basemodel"
      modules: [ "all" ]
      # currently the options of value are as follows:
      #   1> "all": select all hyperparameters in the leaderboard;
      #   2> hyperparameters in the leaderboard, e.g., "momentum"
      hyperparameters: [ "all" ]
      # currently the options of value are as follows:
      #   1> "all": select all metrics in the leaderboard;
      #   2> metrics in the leaderboard, e.g., "f1_score"
      metrics: [ "f1_score", "samples_transfer_ratio" ]

    # save mode of selected and all dataitems in workspace `./rank` ; string type;
    # currently the options of value are as follows:
    #  1> "selected_and_all": save selected and all dataitems;
    #  2> "selected_only": save selected dataitems;
    save_mode: "selected_and_all"
```