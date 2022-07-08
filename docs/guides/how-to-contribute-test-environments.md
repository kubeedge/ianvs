# How to Contribute Test Environments


## Overall contribution workflow

1. Apply for a topic.   
   Once you have new idea about test environment, you can apply for a topic to discuss it on [SIG AI weekly meeting](http://github.com/kubeedge/ianvs.git).
2. Submit proposal.
   After the idea is fully discussed, the former proposal PR is needed to submit to [Ianvs repository](http://github.com/kubeedge/ianvs.git).
3. Fix proposal review comments.  
   If other Ianvs maintainer leave review comments to the PR, you need fix them and get at least 2 reviewers' `/lgtm`, and 1 approver's `/approve`.
4. Submit code.
   Then you can implement your code, and good code style is encouraged.
5. Fix code review comments.  
   Besides the merge requirements of proposal,  CI passing is needed before review in this step.


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



## Add a new test environment

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