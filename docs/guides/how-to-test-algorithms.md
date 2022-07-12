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
  # dataset configuration
  dataset:
    # the url address of train dataset index; string type;
    train_url: "/ianvs/dataset/train_data/index.txt"
    # the url address of test dataset index; string type;
    test_url: "/ianvs/dataset/test_data/index.txt"

  # model eval configuration of incremental learning;
  model_eval:
    # metric used for model evaluation
    model_metric:
      # metric name; string type;
      name: "f1_score"
      # the url address of python file
      url: "./examples/pcb-aoi/incremental_learning_bench/testenv/f1_score.py"

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
      url: "./examples/pcb-aoi/incremental_learning_bench/testenv/f1_score.py"
    - name: "samples_transfer_ratio"

  # incremental rounds setting for incremental learning paradigm.; int type; default value is 2;
  incremental_rounds: 2
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
For this new algorithm in `ClassType.GENERAL`, the code in the algorithm file is as follows: 

```python

@ClassFactory.register(ClassType.GENERAL, alias="FPN")
class BaseModel:

    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """

        self.has_fast_rcnn_predict = False

        self._init_tf_graph()

        self.temp_dir = tempfile.mkdtemp()
        if not os.path.isdir(self.temp_dir):
            mkdir(self.temp_dir)

        os.environ["MODEL_NAME"] = "model.zip"
        cfgs.LR = kwargs.get("learning_rate", 0.0001)
        cfgs.MOMENTUM = kwargs.get("momentum", 0.9)
        cfgs.MAX_ITERATION = kwargs.get("max_iteration", 5)

    def train(self, train_data, valid_data=None, **kwargs):

        if train_data is None or train_data.x is None or train_data.y is None:
            raise Exception("Train data is None.")

        with tf.Graph().as_default():

            img_name_batch, train_data, gtboxes_and_label_batch, num_objects_batch, data_num = \
                next_batch_for_tasks(
                    (train_data.x, train_data.y),
                    dataset_name=cfgs.DATASET_NAME,
                    batch_size=cfgs.BATCH_SIZE,
                    shortside_len=cfgs.SHORT_SIDE_LEN,
                    is_training=True,
                    save_name="train"
                )

            # ... ...
            # several lines are omitted here. 

        return self.checkpoint_path

    def save(self, model_path):
        if not model_path:
            raise Exception("model path is None.")

        model_dir, model_name = os.path.split(self.checkpoint_path)
        models = [model for model in os.listdir(model_dir) if model_name in model]

        if os.path.splitext(model_path)[-1] != ".zip":
            model_path = os.path.join(model_path, "model.zip")

        if not os.path.isdir(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))

        with zipfile.ZipFile(model_path, "w") as f:
            for model_file in models:
                model_file_path = os.path.join(model_dir, model_file)
                f.write(model_file_path, model_file, compress_type=zipfile.ZIP_DEFLATED)

        return model_path

    def predict(self, data, input_shape=None, **kwargs):
        if data is None:
            raise Exception("Predict data is None")

        inference_output_dir = os.getenv("RESULT_SAVED_URL")

        with self.tf_graph.as_default():
            if not self.has_fast_rcnn_predict:
                self._fast_rcnn_predict()
                self.has_fast_rcnn_predict = True

            restorer = self._get_restorer()

            config = tf.ConfigProto()
            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            with tf.Session(config=config) as sess:
                sess.run(init_op)

        # ... ...
        # several lines are omitted here. 

        return predict_dict

    def load(self, model_url=None):
        if model_url:
            model_dir = os.path.split(model_url)[0]
            with zipfile.ZipFile(model_url, "r") as f:
                f.extractall(path=model_dir)
                ckpt_name = os.path.basename(f.namelist()[0])
                index = ckpt_name.find("ckpt")
                ckpt_name = ckpt_name[:index + 4]
            self.checkpoint_path = os.path.join(model_dir, ckpt_name)

        else:
            raise Exception(f"model url is None")

        return self.checkpoint_path

    def evaluate(self, data, model_path, **kwargs):
        if data is None or data.x is None or data.y is None:
            raise Exception("Prediction data is None")

        self.load(model_path)
        predict_dict = self.predict(data.x)
        metric_name, metric_func = kwargs.get("metric")
        if callable(metric_func):
            return {"f1_score": metric_func(data.y, predict_dict)}
        else:
            raise Exception(f"not found model metric func(name={metric_name}) in model eval phase")
```

With the above algorithm interface, one may develop the targeted algorithm of FPN as usual in the same algorithm file. 
The ``FPN_TensorFlow`` is also open sourced. For those interested in ``FPN_TensorFlow``, an example implementation is available [here](https://github.com/DetectionTeamUCAS/FPN_Tensorflow) and extended with the algorithm inferface [here](https://github.com/kubeedge-sedna/FPN_Tensorflow).

Then we can fill the ``algorithm.yaml``: 
``` yaml
algorithm:
  # paradigm type; string type;
  # currently the options of value are as follows:
  #   1> "singletasklearning"
  #   2> "incrementallearning"
  paradigm_type: "incrementallearning"
  incremental_learning_data_setting:
    # ratio of training dataset; float type;
    # the default value is 0.8.
    train_ratio: 0.8
    # the method of splitting dataset; string type; optional;
    # currently the options of value are as follows:
    #   1> "default": the dataset is evenly divided based train_ratio;
    splitting_method: "default"
  # the url address of initial model for model pre-training; string url;
  initial_model_url: "/ianvs/initial_model/model.zip"

  # algorithm module configuration in the paradigm; list type;
  modules:
    # type of algorithm module; string type;
    # currently the options of value are as follows:
    #   1> "basemodel": contains important interfaces such as train, eval, predict and more; required module;
    - type: "basemodel"
      # name of python module; string type;
      # example: basemodel.py has BaseModel module that the alias is "FPN" for this benchmarking;
      name: "FPN"
      # the url address of python module; string type;
      url: "./examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn/basemodel.py"

      # hyperparameters configuration for the python module; list type;
      hyperparameters:
        # name of the hyperparameter; string type;
        - momentum:
            # values of the hyperparameter; list type;
            # types of the value are string/int/float/boolean/list/dictionary
            values:
              - 0.95
              - 0.5
        - learning_rate:
            values:
              - 0.1
      #  2> "hard_example_mining": check hard example when predict ; optional module;
    - type: "hard_example_mining"
      # name of python module; string type;
      name: "IBT"
      # the url address of python module; string type;
      url: "./examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn/hard_example_mining.py"
      # hyperparameters configuration for the python module; list type;
      hyperparameters:
        # name of the hyperparameter; string type;
        # threshold of image; value is [0, 1]
        - threshold_img:
            values:
              - 0.9
        # predict box of image; value is [0, 1]
        - threshold_box:
            values:
              - 0.9
```


The URL address of this algorithm then should be filled in the configuration file of ``benchmarkingJob.yaml`` in the following Step 3. Two examples are as follows: 
``` yaml
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
```

or 

``` yaml
  # the configuration of test object
  test_object:
    # test type; string type;
    # currently the option of value is "algorithms",the others will be added in succession.
    type: "algorithms"
    # test algorithm configuration files; list type;
    algorithms:
      # algorithm name; string type;
      - name: "fpn_singletask_learning"
        # the url address of test algorithm configuration file; string type;
        # the file format supports yaml/yml;
        url: "./examples/pcb-aoi/singletask_learning_bench/testalgorithms/fpn/fpn_algorithm.yaml"
```

## Step 3. ianvs Configuration

Now we comes to the final configuration on ``benchmarkingJob.yaml`` before running ianvs. 

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
