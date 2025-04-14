# How to contribute test environments

## Overall contribution workflow

1. Apply for a topic.
   Once you have a new idea about the test environment, you can apply for a topic to discuss it on [SIG AI weekly meeting](http://github.com/kubeedge/ianvs.git).
2. Submit proposal.
   After the idea is fully discussed, the former proposal PR is needed to submit to the [Ianvs repository](http://github.com/kubeedge/ianvs.git).
3. Fix proposal review comments.  
   If other Ianvs maintainers leave review comments to the PR, you need to fix them and get at least 2 reviewers' `/lgtm`, and 1 approver's `/approve`.
4. Submit code.
   Then you can implement your code, and a good code style is encouraged.
5. Fix code review comments.  
   Besides the merge requirements of the proposal, CI passing is needed before reviewing this step.

The following is a typical testenv:

```yaml
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

It can be found that for a test, we need to set up the three fields:

- dataset
- model_eval
- metrics

That means, if you want to test on a different dataset, different model, or different metrics, you need a new test environment.

## Add a new test environment

Please refer to the examples directory, [cloud-edge-collaborative-inference-for-llm](https://github.com/kubeedge/ianvs/tree/main/examples/cloud-edge-collaborative-inference-for-llm) is a scenario for testing.
We can regard it as a subject for a student that needs to take an exam, the test env is like an examination paper,
and the test job is like the student.

For a subject `cloud-edge-collaborative-inference-for-llm`, a new examination paper could be added to the subdirectory, on the same level as a `benchmarking job`.

The detailed steps could be the following:

1. Copy `benchmarking job` and name `benchmarking job_2` or any other intuitive name.
2. Add new algorithms to test algorithms, or Keep the useful algorithm. It can refer to contribute algorithm section to develop your own algorithm.
3. Copy testenv/testnev.yaml, and modify it based on what you need to test, with different datasets, models, metrics, and so on.

If all things have been done, and you think that would be a nice "examination paper", you can create PR to ianvs, to publish your paper.

Interested "students" from our community will take the exam.

[contribute algorithm]: how-to-contribute-algorithms.md