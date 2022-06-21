[Quick Start]: ./quick-start.md  
[Links of scenarios]: ../proposals/scenarios/
[the PCB-AoI public dataset]: https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi
[details of PCB-AoI dataset]: ../proposals/scenarios/industrial-defect-detection/pcb-aoi.md

# How to test algorithms with Ianvs

With Ianvs installed and related environment prepared, an algorithm developer is then able to test his/her own targeted algorithm using the following steps. 

Note that:
- If you are testing an algorithm summitted in Ianvs repository, e.g., FPN for single task learning, the test environment and the test case are both ready to use and you can directly refer to [Quick Start]. 
- Otherwise, if the user has a test algorithm which is new to Ianvs repository, i.e., the test environment and the test case are not ready for the targeted algorithm, you might test the algorithm in Ianvs following the next steps from scratch.

## Step 1. Test Environment Preparation
  
First, the user need to prepare the dataset according to the targeted scenario, from source links (e.g., from Kaggle) provided by Ianvs. Scenarios with dataset are  available [Links of scenarios]. As an example in this document, we are using [the PCB-AoI Public Dataset] released by KubeEdge SIG AI members on Kaggle. See [details of PCB-AoI dataset] for more information of this dataset. 

You might wonder why not put the dataset on Github repository of Ianvs: Datasets can be large. To avoid over-size projects in the Github repository of Ianvs, the Ianvs code base do not include origin datasets and developers might want to download uneeded datasets. 

The URL address of this dataset then should be filled in the configuration file ``testenv.yaml``. 

``` yaml
# testenv.yaml
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

The URL address of this test environment, i.e., testenv.yaml, then should be filled in the configuration file in the following Step 3. For example,  
``` yaml
# benchmarkingJob.yaml
  testenv: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testenv/testenv.yaml"
```

## Step 2. Test Case Preparation

Note that the tested algorithm should follow the ianvs interface to ensure functional benchmarking.
That is, when a new algorithm is needed for testing, it should to be extended based on the basic classes, i.e., `class_factory.py`. 
The class factory helps to make the algorithm pluggable in Ianvs 
and two classes are defined in `class_factory.py`, namely `ClassType` and `ClassFactory`. 
`ClassFactory` can register the modules you want to reuse through decorators. 
The the user may develop the targeted algorithm as usual using the algorithm interface in class factory. 

Currently, Ianvs is using the `class_factory.py` defined in KubeEdge SIG AI ([source link](https://github.com/kubeedge/sedna/blob/main/lib/sedna/common/class_factory.py)). If you want to contribute a new type of modules to KubeEdge SIG AI, i.e., a new classtype, please refer to the guide of [how to contribute algorithms](./how-to-contribute-algorithms.md).


### Example 1. Testing a hard-example-mining algorithm in incremental learning

As the first example, we describe how to test an algorithm `Threshold-based-HEM` for HEM (Hard Example Mining) module in incremental learning. 
For this new algorithm in `ClassType.HEM`, the code in the algorithm file is as follows:

```python
@ClassFactory.register(ClassType.HEM, alias="Threshold-based-HEM")
class ThresholdFilter(BaseFilter, abc.ABC):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = float(threshold)

    def __call__(self, infer_result=None):
        return Threshold-based-HEM(infer_result)
```

With the above algorithm interface, one may develop the targeted algorithm as usual in the same algorithm file:

```python
def Threshold-based-HEM(infer_result=None):
    # if invalid input, return False
    if not (infer_result
                and all(map(lambda x: len(x) > 4, infer_result))):
            return False

        image_score = 0

        for bbox in infer_result:
            image_score += bbox[4]

        average_score = image_score / (len(infer_result) or 1)
        return average_score < self.threshold
```

### Example 2. Testing a neural-network-based modeling algorithm in incremental learning

As the second example, we describe how to test a neural network `FPN` for HEM (Hard Example Mining) module in incremental learning. 
For this new algorithm in `ClassType.HEM`, the code in the algorithm file is as follows: 

``` python
from FPN_TensorFlow.interface import Estimator as Model

def parse_kwargs(func, **kwargs):
    """ get valid parameters in kwargs """
    if not callable(func):
        return kwargs
    need_kw = getfullargspec(func)
    if need_kw.varkw == 'kwargs':
        return kwargs
    return {k: v for k, v in kwargs.items() if k in need_kw.args}

@ClassFactory.register(ClassType.GENERAL, "estimator")
class BaseModel:
    def __init__(self, **kwargs):
        varkw = parse_kwargs(Model, **kwargs)
        self.model = Model(**varkw)

    def train(self, train_data, valid_data=None, **kwargs):
        return self.model.train(train_data, **kwargs)

    def predict(self, data, **kwargs):
        # data -> image urls
        return self.model.predict(data, **kwargs)

    def load(self, model_url):
        self.model.load(model_url)

    def save(self, model_path):
        return self.model.save(model_path)

    def evaluate(self, data, **kwargs):
        return self.model.evaluate(data, **kwargs)
```

With the above algorithm interface, one may develop the targeted algorithm of FPN as usual in the same algorithm file. 
The ``FPN_TensorFlow`` is also open sourced. For those interested in ``FPN_TensorFlow``, an example implementation is available [here](https://github.com/DetectionTeamUCAS/FPN_Tensorflow) and extended with the algorithm inferface [here](https://github.com/kubeedge-sedna/FPN_Tensorflow).

Then we can fill the ``algorithm.yaml``: 
``` yaml
algorithm:
  paradigm: "incrementallearning"
  dataset_train_ratio: 0.8
  initial_model_url: "/ianvs/pcb-aoi/initial_model/model.zip"
  modules:
    - kind: "basemodel"
      name: "estimator"
      url: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testalgorithms/fpn_incremental_learning/basemodel.py"
      hyperparameters:
        - other_hyperparameters:
            values:
              - "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testalgorithms/fpn_incremental_learning/fpn_hyperparameter.yaml"
```


The URL address of this algorithm then should be filled in the configuration file of ``benchmarkingJob.yaml`` in the following Step 3. Two examples are as follows: 
``` yaml
  algorithms:
    - name: "fpn_singletask_learning"
      url: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testalgorithms/fpn_singletask_learning/fpn_algorithm.yaml"
```

or 

``` yaml
  algorithms:
    - name: "fpn_incremental_learning"
      url: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testalgorithms/fpn_incremental_learning/fpn_algorithm.yaml"
```

## Step 3. ianvs Configuration

Now we comes to the final configuration on ``benchmarkingJob.yaml'' before running ianvs. 

First, the user can configure the workspace to reserve the output of tests. 
``` yaml
# benchmarkingJob.yaml
  workspace: "/ianvs/pcb-aoi/workspace/"
```

Then, the user fill in the test environment and algorithm configured in previous steps. 
``` yaml
# benchmarkingJob.yaml
  testenv: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testenv/testenv.yaml"
```
``` yaml
  algorithms:
    - name: "fpn_incremental_learning"
      url: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testalgorithms/fpn_incremental_learning/fpn_algorithm.yaml"
```

As the final leaderboard, the user can configure how to rank the leaderboard with the specific metric and order. 
``` yaml
# benchmarkingJob.yaml
    rank:
        sort_by: [ { "f1_score": "descend" } ]
```

There are quite a few possible dataitems in the leaderboard. Not all of them can be shown simultaneously on the screen. In the leaderboard, we provide the ``selected_only`` mode for the user to configure what is shown or is not shown. The user can add his/her interested dataitems in terms of ``paradigms``, ``modules``, ``hyperparameters`` and ``metrics``, so that the selected columns will be shown.
``` yaml
    visualization:
      mode: "selected_only"
      method: "print_table"

    selected_dataitem:
      paradigms: [ "all" ]
      modules: [ "all" ]
      hyperparameters: [ "all" ]
      metrics: [ "f1_score" ]

    save_mode: "selected_and_all"
```


## Step 4. Execution and Presentation

Finally, the user can run ianvs for benchmarking. 

The benchmarking result of the targeted algorithms will be shown after evaluation is done. Leaderboard examples can be found [here](../proposals/leaderboards).
