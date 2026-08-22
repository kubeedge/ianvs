# How to config algorithm

Lets take the example of [cloud-edge-collaborative-inference-for-llm](../proposals/scenarios/cloud-edge-collaborative-inference-for-llm/mmlu-5-shot.md) scenario and understand how algorithm developer is able to test his/her own targeted algorithm and configs the algorithm using the following configuration.

## The configuration of algorithm

### Model Configuration

The models are configured in `examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/test_queryrouting.yaml`.

In the configuration file, there are two models available for configuration: `EdgeModel` and `CloudModel`.

#### EdgeModel Configuration

The `EdgeModel` is the model that will be deployed on your local machine, supporting `huggingface` and `vllm` as serving backends.

For `EdgeModel`, the open parameters are:

| Parameter Name         | Type  | Description                                                  | Default                  |
| ---------------------- | ----- | ------------------------------------------------------------ | ------------------------ |
| model                  | str   | model name                                                   | Qwen/Qwen2-1.5B-Instruct |
| backend                | str   | model serving framework                                      | huggingface              |
| temperature            | float | What sampling temperature to use, between 0 and 2            | 0.8                      |
| top_p                  | float | nucleus sampling parameter                                   | 0.8                      |
| max_tokens             | int   | The maximum number of tokens that can be generated in the chat completion | 512                      |
| repetition_penalty     | float | The parameter for repetition penalty                         | 1.05                     |
| tensor_parallel_size   | int   | The size of tensor parallelism (Used for vLLM)               | 1                        |
| gpu_memory_utilization | float | The percentage of GPU memory utilization (Used for vLLM)     | 0.9                      |

#### CloudModel Configuration

The `CloudModel` represents the model on cloud, it will call LLM API via OpenAI API format.

For `CloudModel`, the open parameters are:

| Parameter Name     | Type | Description                                                  | Default     |
| ------------------ | ---- | ------------------------------------------------------------ | ----------- |
| model              | str  | model name                                                   | gpt-4o-mini |
| temperature        | float  | What sampling temperature to use, between 0 and 2            | 0.8         |
| top_p              | float  | nucleus sampling parameter                                   | 0.8         |
| max_tokens         | int  | The maximum number of tokens that can be generated in the chat completion | 512         |
| repetition_penalty | float  | The parameter for repetition penalty                         | 1.05        |

#### Router Configuration

Router is a component that routes the query to the edge or cloud model. The router is configured by `hard_example_mining` in `examples/cloud-edge-collaborative-inference-for-llm/testrouters/query-routing/test_queryrouting.yaml`.

Currently, supported routers include:

| Router Type  | Description                                                  | Parameters       |
| ------------ | ------------------------------------------------------------ | ---------------- |
| EdgeOnly     | Route all queries to the edge model.                         | -                |
| CloudOnly    | Route all queries to the cloud model.                        | -                |
| OracleRouter | Optimal Router         |         |
| BERTRouter   | Use a BERT classifier to route the query to the edge or cloud model. | model, threshold |
| RandomRouter | Route the query to the edge or cloud model randomly.         | threshold        |

You can modify the `router` parameter in `test_queryrouting.yaml` to select the router you want to use.

For BERT router, you can use [routellm/bert](https://huggingface.co/routellm/bert) or [routellm/bert_mmlu_augmented](https://huggingface.co/routellm/bert_mmlu_augmented) or your own BERT model.

#### Data Processor Configuration

The Data Processor allows you to customize your own data format after the dataset gets loaded.

Currently, supported routers include:

| Data Processor  | Description                                                  | Parameters       |
| ------------ | ------------------------------------------------------------ | ---------------- |
| OracleRouterDatasetProcessor     |  Expose `gold` label to OracleRouter                      |   -         |

## Show example

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
            # For reproducible results, the temperature should be set to 0;
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
            # Whether to use response cache; boolean type;
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
