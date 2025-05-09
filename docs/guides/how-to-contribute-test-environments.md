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
    train_index: "./dataset/train_data/index.txt"
    # the url address of test dataset index; string type;
    test_index: "./dataset/test_data/index.txt"

  # model eval configuration of incremental learning;
  model_eval:
    # metric used for model evaluation
    model_metric:
      # metric name; string type;
      name: "f1_score"
      # the url address of python file
      url: "./examples/pcb-aoi/incremental_learning_bench/fault_detection/testenv/f1_score.py"

    # condition of triggering inference model to update
    # threshold of the condition; types are float/int
    threshold: 0.01
    # operator of the condition; string type;
    # values are ">=", ">", "<=", "<" and "=";
    operator: ">="

  # metrics configuration for test case's evaluation; list type;
  metrics:
      # metric name; string type;
    - name: "f1_score"
      # the url address of python file
      url: "./examples/pcb-aoi/incremental_learning_bench/fault_detection/testenv/f1_score.py"
    - name: "samples_transfer_ratio"

  # incremental rounds setting for incremental learning paradigm.; int type; default value is 2;
  incremental_rounds: 2
```

It can be found that for a test, we need to set up the three fields:

- dataset
- model_eval
- metrics

That means, if you want to test on a different dataset, different model, or different metrics, you need a new test environment.

## Add a new test environment

Please refer to the examples directory, [pcb-aoi](https://github.com/kubeedge/ianvs/tree/main/examples/pcb-aoi) is a scenario for testing.
We can regard it as a subject for a student that needs to take an exam, the test env is like an examination paper,
and the test job is like the student.

For a subject `pcb-aoi`, a new examination paper could be added to the subdirectory, on the same level as a `benchmarking job`.
The detailed steps could be the following:

1. Copy `benchmarking job` and name `benchmarking job_2` or any other intuitive name.
2. Add new algorithms to test algorithms, or Keep the useful algorithm. It can refer to contribute algorithm section to develop your own algorithm.
3. Copy testenv/testnev.yaml, and modify it based on what you need to test, with different datasets, models, metrics, and so on.

If all things have been done, and you think that would be a nice "examination paper", you can create PR to ianvs, to publish your paper.

Interested "students" from our community will take the exam.

[contribute algorithm]: how-to-contribute-algorithms.md