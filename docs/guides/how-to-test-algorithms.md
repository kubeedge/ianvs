# How to test algorithms with Ianvs

With Ianvs installed and related environment prepared, an algorithm developer is then able to test his/her own targeted algorithm using the following steps. 

## Step 1. Test Environment Preparation
  
First, we prepare the dataset according to the targeted scenario. In this document, we are using [the PCB-AoI dataset](https://www.kaggle.com/datasets/kubeedgeianvs/pcb-aoi) released by KubeEdge SIG AI members on Kaggle. 
  
Why not put the dataset on Github: Datasets can be large. To avoid over-size projects in the Github repository of Ianvs, the Ianvs code base do not include origin datasets and developers might want to download datasets from source links (e.g., from Kaggle) provided by Ianvs. 

The URL address of this dataset then should be filled in the configuration file in the following Step 3. 

``` yaml
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

The URL address of this test environment, i.e., testenv.yaml, then should be filled in the configuration file in the following Step 3. 

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


The URL address of this algorithm then should be filled in the configuration file in the following Step 3. 


## Step 3. ianvs Configuration

Fill configuration files for ianvs

测试工作空间，保存本地多个测试工作的输出， 字符串类型，可选，默认值是 "./workspace/"
``` yaml
  workspace: "/ianvs/pcb-aoi/workspace/"
```

测试环境配置（考卷）, 字符串类型，必选， 默认值是 “”；
``` yaml
  testenv: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testenv/testenv.yaml"
```

测试算法列表
``` yaml
  algorithms:
    - name: "fpn_singletask_learning"
      url: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testalgorithms/fpn_singletask_learning/fpn_algorithm.yaml"
  #    - name: "fpn_incremental_learning"
  #      url: "/home/yj/ianvs/examples/pcb-aoi/benchmarkingjob/testalgorithms/fpn_incremental_learning/fpn_algorithm.yaml"
```

排行榜的排序规则，列表类型，可选，默认值是 []；
列表的元素详解：字典类型，其中key是ianvs评估的指标计算方法，字符串类型，
              value是排序方式，当前支持 “ascend" 和 ”descend" ；
排行榜的排序优先级按照列表中元素从前往后依次排列而定，比如 "sampe"的优先级要高于"max_error_rate"；

可视化模式，字符串类型，可选，当前支持 "off"和 "selected_only"，默认值是 "selected_only"；
可视化方法，字符串类型，可选，当前支持"print_table"，默认值是 "print_table"；

白名单，通过范式、基础模型和超参列表来筛选元素可视化，字典类型，可选， 默认值是 {}，
范式名字，列表类型，可选，当前支持"lifelonglearning"和"all", 默认值是 ""；
基础模型名字，列表类型，可选，当前支持"all", 默认值是 ""；
超参列表，列表类型，可选，当前支持"all"及相关超参数，默认值是[]；
指标列表，列表类型，可选，当前支持"all"，默认值是[]；
保存模式，列表类型，可选，当前支持 "off", "selected_only", "selected_and_all"，默认值是 "selected_and_all"；
``` yaml
    rank:
        sort_by: [ { "f1_score": "descend" } ]

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

Run the executable file of ianvs for benchmarking

View the benchmarking result of the targeted algorithms