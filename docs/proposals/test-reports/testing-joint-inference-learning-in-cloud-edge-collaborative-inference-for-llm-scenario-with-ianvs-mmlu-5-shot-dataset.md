# Testing Joint Inference Learning in Cloud Edge Collaborative Inference for LLM Scenario with Ianvs-MMLU-5-shot dataset

The Deatils of Cloud Edge Collaborative Inference for LLM Scenario can be found [here](../scenarios/cloud-edge-collaborative-inference-for-llm/mmlu-5-shot.md) and the details of query-routing algorithm can be found [here](../algorithms/joint-inference/query-routing.md). 

## Benchmark Settings

Key settings of the `test environment` of `cloud-edge-collaborative-inference-for-llm` are as follows:

``` yaml
# testenv.yaml
testenv:
  # dataset configuration
  dataset:
    # the url address of train dataset index; string type;
    train_data: "./dataset/mmlu-5-shot/train_data/data.json"
    # the url address of test dataset index; string type;
    test_data_info: "./dataset/mmlu-5-shot/test_data/metadata.json"

  # metrics configuration for test case's evaluation; list type;
  metrics:
      # metric name; string type;
    - name: "Accuracy"
      # the url address of python file
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/accuracy.py"

    - name: "Edge Ratio"
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/edge_ratio.py"

    - name: "Cloud Prompt Tokens"
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/cloud_prompt_tokens.py"

    - name: "Cloud Completion Tokens"
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/cloud_completion_tokens.py"

    - name: "Edge Prompt Tokens"
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/edge_prompt_tokens.py"

    - name: "Edge Completion Tokens"
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/edge_completion_tokens.py"

    - name: "Time to First Token"
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/time_to_first_token.py"

    - name: "Throughput"
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/throughput.py"

    - name: "Internal Token Latency"
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/internal_token_latency.py"
```

Key settings of the `Query-Routing` algorithm for `cloud-edge-collaborative-inference-for-llm` are as follows:

```yaml
# test_queryrouting.yaml
algorithm:
  # paradigm name; string type;
  paradigm_type: "jointinference"

  # algorithm module configuration in the paradigm; list type;
  modules:
    # kind of algorithm module; string type;
    - type: "dataset_processor"
      # name of custom dataset processor; string type;
      name: "OracleRouterDatasetProcessor"
      # the url address of custom dataset processor; string type;
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/data_processor.py"

    - type: "edgemodel"
      # name of edge model module; string type;
      name: "EdgeModel"
      # the url address of edge model module; string type;
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/edge_model.py"

      hyperparameters:
      # name of the hyperparameter; string type;
        - model:
            values:
              - "Qwen/Qwen2.5-1.5B-Instruct"
              - "Qwen/Qwen2.5-3B-Instruct"
              - "Qwen/Qwen2.5-7B-Instruct"
        - backend:
            # backend; string type;
            # currently the options of value are as follows:
            #  1> "huggingface": transformers backend;
            #  2> "vllm": vLLM backend;
            #  3> "api": OpenAI API backend;
            values:
              - "vllm"
        - temperature:
            # What sampling temperature to use, between 0 and 2; float type;
            # For reproducable results, the temperature should be set to 0;
            values:
              - 0
        - top_p:
            # nucleus sampling parameter; float type;
            values:
              - 0.8
        -  max_tokens:
            # The maximum number of tokens that can be generated in the chat completion; int type;
            values:
              - 512
        -  repetition_penalty:
            # The parameter for repetition penalty; float type;
            values:
              - 1.05
        -  tensor_parallel_size:
            # The size of tensor parallelism (Used for vLLM)
            values:
              - 4
        -  gpu_memory_utilization:
            # The percentage of GPU memory utilization (Used for vLLM)
            values:
              - 0.9
        -  use_cache:
            # Whether to use reponse cache; boolean type;
            values:
              - true

    - type: "cloudmodel"
      # name of python module; string type;
      name: "CloudModel"
      # the url address of python module; string type;
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/cloud_model.py"

      hyperparameters:
        # name of the hyperparameter; string type;
        - model:
            values:
              - "gpt-4o-mini"
        - temperature:
            values:
              - 0
        - top_p:
            values:
              - 0.8
        -  max_tokens:
            values:
              - 512
        -  repetition_penalty:
            values:
              - 1.05
        -  use_cache:
            values:
              - true

    - type: "hard_example_mining"
      # name of Router module; string type;
      # BERTRouter, EdgeOnly, CloudOnly, RandomRouter, OracleRouter
      name: "EdgeOnly"
      # the url address of python module; string type;
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/hard_sample_mining.py"
```

The final `benchmarkingjob.yaml` for `cloud-edge-collaborative-inference-for-llm` looks like this:

```yaml
# benchmarkingjob.yaml
benchmarkingjob:
  # job name of bechmarking; string type;
  name: "benchmarkingjob"
  # the url address of job workspace that will reserve the output of tests; string type;
  # "~/" cannot be identified, so must be relative path or absoulute path
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

## Benchmark Result

We released the leaderboard [here](../leaderboards/leaderboard-in-cloud-edge-collaborative-inference-for-llm/leaderboard-of-cloud-edge-collaborative-inference-for-llm.md).