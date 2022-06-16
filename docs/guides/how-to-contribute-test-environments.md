# How to Contribute Test Environments


## When a new test environment is needed?

The following is a typical testenv: 
```yaml
testenv:
  dataset:  
    url: "/ianvs/pcb-aoi/dataset/trainData.txt"
    train_ratio: 0.8
    splitting_method: "default"
  model_eval:
    model_metric:
      name: "f1_score"
      url: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testenv/f1_score.py"
    threshold: 0
    operator: ">="
  metrics:
    - name: "f1_score"
      url: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testenv/f1_score.py"
  incremental_rounds: 1
```
It can be found that, for a test we need to setup the three fields: 
- dataset
- model_eval
- metrics

That means, if you want to test on different dataset, different model or different metrics, you need a new test environment.



## How to add a new test environment?

Please refer to the examples directory, [pcb-aoi] is a scenario for testing.
We can regard it as a subject for a student that need to take an exam, the test env is like examination paper,
and the test job is like the student.

For a subject `pcb-aoi`, a new examination paper could be added to the subdirectory, on the same level as `benchmarkingjob`.
The detailed steps could be the following:
1. Copy `benchmarkingjob` and named `benchmarkingjob_2` or any other intuitive name.
2. Add new algorithm to `testalgorithms`, or Keep the useful algorithm. It can refer to [contribute algorithm] section to develop your own algorithm.   
3. Copy `testenv/testnev.yaml`, and modify it based on what you need to test, with different dataset, model, metrics and so on.

If all things have been done, and you think that would be a nice "examination paper", you can create PR to ianvs, to publish your paper.

Interested "student" from our comunity will take the exam.








[pcb-aoi]: ../../examples/pcb-aoi
[contribute algorithm]: how-to-contribute-algorithms.md