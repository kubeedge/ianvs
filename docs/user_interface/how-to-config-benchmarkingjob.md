# How to config benchmarkingjob

Lets take the example of [cloud-edge-collaborative-inference-for-llm](../proposals/scenarios/cloud-edge-collaborative-inference-for-llm/mmlu-5-shot.md) scenario and understand how algorithm developer is able to test his/her own targeted algorithm and configs the benchmarkingjob using the following configuration.

## The configuration of benchmarkingjob

| Property | Required | Description |
|----------|----------|-------------|
|name|yes|Job name of benchmarking; Type: string|
|workspace|no|The url address of job workspace that will reserve the output of tests; Type: string; Default value: `./workspace`|
|testenv|yes|The url address of test environment configuration file; Type: string; Value Constraint: The file format supports yaml/yml.|
|test_object|yes|The configuration of test_object|
|rank|yes|The configuration of ranking leaderboard|

For example:

```yaml
benchmarkingjob:
  # job name of benchmarking; string type;
  name: "benchmarkingjob"
  # the url address of job workspace that will reserve the output of tests; string type;
  # default value: "./workspace"
  workspace: "./workspace-mmlu"

  # the url address of test environment configuration file; string type;
  # the file format supports yaml/yml;
  testenv: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml"
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
|algorithms|no|Test algorithm configuration; Type: list|

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
    - name: "query-routing"
      # the url address of test algorithm configuration file; string type;
      # the file format supports yaml/yml;
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml"
```

### The configuration of rank

| Property | Required | Description |
|----------|----------|-------------|
|sort_by|yes|Rank leaderboard with metric of test case's evaluation and order; Type: list; Value Constraint: The sorting priority is based on the sequence of metrics in the list from front to back.|
|visualization|yes|The configuration of visualization|
|selected_dataitem|yes|The configuration of selected_dataitem; The user can add his/her interested dataitems in terms of "paradigms", "modules", "hyperparameters" and "metrics", so that the selected columns will be shown.|
|save_mode|yes|save mode of selected and all dataitems in workspace `./rank`; Type: string; Value Constraint: Currently the options of value are as follows: 1> "selected_and_all": save selected and all dataitems. 2> "selected_only": save selected dataitems.|

For example:

```yaml
# the configuration of ranking leaderboard
rank:
  # rank leaderboard with metric of test case's evaluation and order ; list type;
  # the sorting priority is based on the sequence of metrics in the list from front to back;
  sort_by: [ { "Accuracy": "descend" } ]
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
  modules: [ "hard_example_mining" ]
  # currently the options of value are as follows:
  #   1> "all": select all hyperparameters in the leaderboard;
  #   2> hyperparameters in the leaderboard, e.g., "momentum"
  hyperparameters: [ "edgemodel-model", "edgemodel-backend", "cloudmodel-model"]
  # currently the options of value are as follows:
  #   1> "all": select all metrics in the leaderboard;
  #   2> metrics in the leaderboard, e.g., "F1_SCORE"
  metrics: ["Accuracy", "Edge Ratio", "Time to First Token", "Throughput", "Internal Token Latency", "Cloud Prompt Tokens", "Cloud Completion Tokens", "Edge Prompt Tokens", "Edge Completion Tokens"]
```

## Show the example

```yaml
# benchmarking.yaml
benchmarkingjob:
  # job name of benchmarking; string type;
  name: "benchmarkingjob"
  # the url address of job workspace that will reserve the output of tests; string type;
  # "~/" cannot be identified, so must be relative path or absolute path
  workspace: "./workspace-mmlu"

  hard_example_mining_mode: "mining-then-inference"

  # the url address of test environment configuration file; string type;
  # the file format supports yaml/yml;
  testenv: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml"

  # the configuration of test object
  test_object:
    # test type; string type;
    # currently the option of value is "algorithms",the others will be added in succession.
    type: "algorithms"
    # test algorithm configuration files; list type;
    algorithms:
      # algorithm name; string type;
      - name: "query-routing"
        # the url address of test algorithm configuration file; string type;
        # the file format supports yaml/yml;
        url: "./examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml"

  # the configuration of ranking leaderboard
  rank:
    # rank leaderboard with metric of test case's evaluation and order ; list type;
    # the sorting priority is based on the sequence of metrics in the list from front to back;
    sort_by: [ { "Accuracy": "descend" } ]

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
      modules: [ "hard_example_mining" ]
      # currently the options of value are as follows:
      #   1> "all": select all hyperparameters in the leaderboard;
      #   2> hyperparameters in the leaderboard, e.g., "momentum"
      hyperparameters: [ "edgemodel-model", "edgemodel-backend", "cloudmodel-model"]
      # currently the options of value are as follows:
      #   1> "all": select all metrics in the leaderboard;
      #   2> metrics in the leaderboard, e.g., "f1_score"
      # metrics: [ "acc" , "edge-rate", "cloud-prompt", "cloud-completion", "edge-prompt", "edge-completion", "input-throughput", "output-throughput", "latency"]
      metrics: ["Accuracy", "Edge Ratio", "Time to First Token", "Throughput", "Internal Token Latency", "Cloud Prompt Tokens", "Cloud Completion Tokens", "Edge Prompt Tokens", "Edge Completion Tokens"]

    # model of save selected and all dataitems in workspace; string type;
    # currently the options of value are as follows:
    #  1> "selected_and_all": save selected and all dataitems;
    #  2> "selected_only": save selected dataitems;
    save_mode: "selected_and_all"
```
