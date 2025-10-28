[Quick Start]: ./quick-start.md
[Links of scenarios]: ../proposals/scenarios/cloud-edge-collaborative-inference-for-llm/mmlu-5-shot.md
[Ianvs-MMLU-5-shot dataset]: https://www.kaggle.com/datasets/kubeedgeianvs/ianvs-mmlu-5shot
[Details of Ianvs-MMLU-5-shot dataset]: ../proposals/scenarios/cloud-edge-collaborative-inference-for-llm/mmlu-5-shot.md

# How to test algorithms with Ianvs

With Ianvs installed and the related environment prepared, an algorithm developer is then able to test his/her own targeted algorithm using the following steps.

**Note that**:

- If you are testing an algorithm submitted in the Ianvs repository, e.g., Query-Routing for Cloud-Edge-Collaborative-Inference-For-LLM, the test environment and the test case are both ready to use and you can directly refer to [Quick Start].
- Otherwise, if the user has a test algorithm that is new to the Ianvs repository, i.e., the test environment and the test case are not ready for the targeted algorithm, you might test the algorithm in Ianvs following the next steps from scratch.

## Step 1. Test Environment Preparation
  
First, the user needs to prepare the dataset according to the targeted scenario, from source links (e.g., from Kaggle or Hugging Face) provided by Ianvs. Scenarios with datasets are available at: [Links of scenarios]. As an example in this document, we are using the [Ianvs-MMLU-5-shot dataset] released by KubeEdge SIG AI members on Hugging Face. See details of [Ianvs-MMLU-5-shot dataset] for more information on this dataset.

You might wonder why not put the dataset on the GitHub repository of Ianvs: Datasets can be large. To avoid over-size projects in the GitHub repository of Ianvs, the Ianvs code base does not include origin datasets and developers might want to download unneeded datasets.
The URL address of this dataset then should be filled in the configuration file testenv.yaml.

The URL address of this dataset then should be filled in the configuration file ``testenv.yaml``.

``` yaml
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
      # other metrics
      ... 
```

The URL address of this test environment, i.e., testenv.yaml, then should be filled in the configuration file in the following Step 3. For example,  

``` yaml
# benchmarkingJob.yaml
testenv: "./examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml"
```

## Step 2. Test Case Preparation

Note that the tested algorithm should follow the ianvs interface to ensure functional benchmarking. That is, when a new algorithm is needed for testing, it should be extended based on the basic classes, i.e., class_factory.py. The class factory helps to make the algorithm pluggable in Ianvs and two classes are defined in class_factory.py, namely `ClassType`, and `ClassFactory`. ClassFactory can register the modules you want to reuse through decorators. The user may develop the targeted algorithm, as usual, using the algorithm interface in the class factory.

Currently, Ianvs is using the `class_factory.py` defined in KubeEdge SIG AI ([source link](https://github.com/kubeedge/sedna/blob/main/lib/sedna/common/class_factory.py)). If you want to contribute a new type of module to KubeEdge SIG AI, i.e., a new classtype, please refer to the guide of [how to contribute algorithms](./how-to-contribute-algorithms.md).

### Example 1: Testing a Query-Routing Algorithm in Joint Inference Learning with Cloud-Edge-Collaborative-Inference-For-LLM scenario

As the first example, we describe how to test an algorithm `Query-Routing` for HEM (Hard Example Mining) module in Joint Inference Learning with cloud-edge-collaborative-inference-for-llm scenario.
For this new algorithm in `ClassType.HEM`, the code in the algorithm file is as follows:

```python
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('BERTFilter', 'EdgeOnlyFilter', 'CloudOnlyFilter',
           'RandomRouterFilter', 'OracleRouterFilter')

@ClassFactory.register(ClassType.HEM, alias="BERTRouter")
class BERTFilter(BaseFilter, abc.ABC):
    """BERTRouter Logic"""
    ...

@ClassFactory.register(ClassType.HEM, alias="EdgeOnly")
class EdgeOnlyFilter(BaseFilter, abc.ABC):
    """EdgeOnly Router Logic"""
    ...

@ClassFactory.register(ClassType.HEM, alias="CloudOnly")
class CloudOnlyFilter(BaseFilter, abc.ABC):
    """CloudOnly Router Logic"""
    ...

@ClassFactory.register(ClassType.HEM, alias="RandomRouter")
class RandomRouterFilter(BaseFilter, abc.ABC):
    """RandomRouter Router Logic"""
    ...

@ClassFactory.register(ClassType.HEM, alias="OracleRouter")
class OracleRouterFilter(BaseFilter, abc.ABC):
  """"OracleRouter Logic"""
    ...
```

With the above algorithm interface, one may develop the targeted algorithm as usual in the same algorithm file:

```python
"""Hard Example Mining Algorithms"""

import abc
import random
from transformers import pipeline
from sedna.common.class_factory import ClassFactory, ClassType
from core.common.log import LOGGER

__all__ = ('BERTFilter', 'EdgeOnlyFilter', 'CloudOnlyFilter',
           'RandomRouterFilter', 'OracleRouterFilter')

class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __init__(self, **kwargs):
        LOGGER.info(f"USING {self.__class__.__name__}")

    def __call__(self, infer_result=None):
        """
        predict function, judge the sample is hard or not.

        Parameters
        ----------
        infer_result : array_like
            prediction result

        Returns
        -------
        is_hard_sample : bool
            `True` means hard sample, `False` means not.
        """
        raise NotImplementedError

    @classmethod
    def data_check(cls, data):
        """Check the data in [0,1]."""
        return 0 <= float(data) <= 1

@ClassFactory.register(ClassType.HEM, alias="BERTRouter")
class BERTFilter(BaseFilter, abc.ABC):
    def __init__(self, **kwargs):
        """Initialize the BERTFilter.

        Parameters
        ----------
        kwargs: dict
            Possible kwargs are:
            - `model`: str, default "routellm/bert". The model to be used.
            - `task`: str, default "text-classification". The task to be used.
            - `max_length`: int, default 512. The maximum length of the input.
        """
        super().__init__(**kwargs)

        self.kwargs = kwargs
        LOGGER.info(kwargs)

        self.model = kwargs.get("model", "routellm/bert")
        self.task = kwargs.get("task", "text-classification")
        self.max_length = kwargs.get("max_length", 512)

        self.classifier = pipeline(self.task, model=self.model, device="cuda")

    def _text_classification_postprocess(self, result):
        """Postprocess the text classification result

        Parameters
        ----------
        result : list
            The result from the classifier. Example:
            [{"label": "LABEL_0", "score": 0.5},
            {"label": "LABEL_1", "score": 0.4},
            {"label": "LABEL_2", "score": 0.1}]

        Returns
        -------
        bool
            `True` means hard sample, `False` means not.
        """

        res = {item["label"]:item["score"] for item in result}
        scaled_score = res["LABEL_0"] / (res["LABEL_0"] + res["LABEL_1"])

        thresold = self.kwargs.get("threshold", 0.5)
        label = "LABEL_0" if scaled_score >= thresold else "LABEL_1"
        return False if label == "LABEL_0" else True

    def _predict(self, data):
        """Predict the data label

        Parameters
        ----------
        data : dict
            See format at BaseLLM's `inference()`.

        Returns
        -------
        bool
            `True` means hard sample, `False` means not.

        Raises
        ------
        NotImplementedError
            If the task is not supported
        """

        if self.task == "text-classification":
            result = self.classifier(data, top_k=None)
            is_hard_sample = self._text_classification_postprocess(result)
        else:
            raise NotImplementedError

        return is_hard_sample

    def _preprocess(self, data):
        """Preprocess the data

        Parameters
        ----------
        data : dict
            See format at BaseLLM's `inference()`.

        Returns
        -------
        str
            query string
        """
        query = data.get("query")
        if "query" in query:
            return query["query"][:self.max_length]
        else:
            return query[:self.max_length]


    def cleanup(self):
        """Release the classifier model
        """
        del self.classifier

    def __call__(self, data=None) -> bool:
        data = self._preprocess(data)
        return self._predict(data)

@ClassFactory.register(ClassType.HEM, alias="EdgeOnly")
class EdgeOnlyFilter(BaseFilter, abc.ABC):
    """Route all queries to edge.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data=None) -> bool:
        return False

@ClassFactory.register(ClassType.HEM, alias="CloudOnly")
class CloudOnlyFilter(BaseFilter, abc.ABC):
    """Route all queries to cloud.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data=None) -> bool:
        return True

@ClassFactory.register(ClassType.HEM, alias="RandomRouter")
class RandomRouterFilter(BaseFilter, abc.ABC):
    """Randomly route the queries to edge or cloud.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.threshold = kwargs.get("threshold", 0)

    def __call__(self, data=None) -> bool:
        return False if random.random() < self.threshold else True

@ClassFactory.register(ClassType.HEM, alias="OracleRouter")
class OracleRouterFilter(BaseFilter, abc.ABC):
    """The Opitmal Router, which routes the queries to edge or cloud based on the models' prediction.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.edge_better = 0
        self.cloud_better = 0
        self.both_right = 0
        self.both_wrong = 0

        self.edge_model = kwargs.get("edgemodel")
        self.cloud_model = kwargs.get("cloudmodel")

    def __call__(self, data=None):
        """Route the query to edge or cloud based on the models' prediction.

        Parameters
        ----------
        data : dict
            See format at BaseLLM's `inference()`.

        Returns
        -------
        bool
            `True` means hard sample, `False` means not.
        """
        gold = data.get("gold", None)

        edge_result = self.edge_model.predict(data).get("prediction")
        cloud_result = self.cloud_model.inference(data).get("prediction")

        both_right = edge_result == gold and cloud_result == gold
        both_wrong = edge_result != gold and cloud_result != gold
        edge_better = edge_result == gold and cloud_result != gold
        cloud_better = edge_result != gold and cloud_result == gold

        if both_right:
            self.both_right +=1
        elif both_wrong:
            self.both_wrong += 1
        elif edge_better:
            self.edge_better += 1
        elif cloud_better:
            self.cloud_better += 1

        if cloud_better:
            # cloud is better than edge, hard sample
            return True
        else:
            # both correct + both wrong + edge_better, easy sample
            return False

    def cleanup(self):
        """Leverage the `cleanup()` interface to print the statistics.
        """
        message = [
            f"OracleRouter Statistics: \n",
            f"Both Wrong: {self.both_wrong},  ",
            f"Both Correct: {self.both_right},  ",
            f"Edge Better: {self.edge_better},  ",
            f"Cloud Better: {self.cloud_better}"
        ]
        LOGGER.info("".join(message))

```

Then we can fill in the ``test_queryrouting.yaml``:

```yaml
algorithm:
  # paradigm name; string type;
  paradigm_type: "jointinference"

  # algorithm module configuration in the paradigm; list type;
  modules:
    # kind of algorithm module; string type;
    - type: "hard_example_mining"
      # name of Router module; string type;
      # BERTRouter, EdgeOnly, CloudOnly, RandomRouter, OracleRouter
      name: "EdgeOnly"
      # the url address of python module; string type;
      url: "./examples/cloud-edge-collaborative-inference-for-llm/testalgorithms/query-routing/hard_sample_mining.py"
```

The URL address of this algorithm then should be filled in the configuration file of ``benchmarkingJob.yaml`` in the following Step 3. Example:- 

```yaml
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
```

### Example 2: Testing a hard-example-mining algorithm in incremental learning

As the second example, we describe how to test an algorithm `Threshold-based-HEM` for HEM (Hard Example Mining) module in incremental learning.
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

### Example 3: Testing a neural-network-based modeling algorithm in incremental learning

As the third example, we describe how to test a neural network `FPN` for HEM (Hard Example Mining) module in incremental learning.
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

class BaseFilter(metaclass=abc.ABCMeta):
    """The base class to define unified interface."""

    def __init__(self, **kwargs):
        LOGGER.info(f"USING {self.__class__.__name__}")

    def __call__(self, infer_result=None):
        """
        predict function, judge the sample is hard or not.

        Parameters
        ----------
        infer_result : array_like
            prediction result

        Returns
        -------
        is_hard_sample : bool
            `True` means hard sample, `False` means not.
        """
        raise NotImplementedError

    @classmethod
    def data_check(cls, data):
        """Check the data in [0,1]."""
        return 0 <= float(data) <= 1
```

With the above algorithm interface, one may develop the targeted algorithm of FPN as usual in the same algorithm file.
The ``FPN_TensorFlow`` is also open sourced. For those interested in ``FPN_TensorFlow``, an example implementation is available [here](https://github.com/DetectionTeamUCAS/FPN_Tensorflow) and extended with the algorithm interface [here](https://github.com/ECIL-EdgeAI/FPN_Tensorflow).

Then we can fill in the ``algorithm.yaml``:

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
  initial_model_url: "./initial_model/model.zip"

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
      url: "./examples/pcb-aoi/incremental_learning_bench/fault_detection/testalgorithms/fpn/basemodel.py"

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
      url: "./examples/pcb-aoi/incremental_learning_bench/fault_detection/testalgorithms/fpn/hard_example_mining.py"
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
        url: "./examples/pcb-aoi/incremental_learning_bench/fault_detection/testalgorithms/fpn/fpn_algorithm.yaml"
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
        url: "./examples/pcb-aoi/singletask_learning_bench/fault_detection/testalgorithms/fpn/fpn_algorithm.yaml"
```

## Step 3. ianvs Configuration

Now we come to the final configuration on ``benchmarkingJob.yaml`` before running ianvs.

First, the user can configure the workspace to reserve the output of tests.

``` yaml
# benchmarkingJob.yaml
  workspace: "./workspace-mmlu"
```

Then, the user fill in the test environment and algorithm configured in previous steps. 

``` yaml
# benchmarkingJob.yaml
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
```

As the final leaderboard, the user can configure how to rank the leaderboard with the specific metric and order.

``` yaml
# benchmarkingJob.yaml
  rank:
      # rank leaderboard with metric of test case's evaluation and order ; list type;
      # the sorting priority is based on the sequence of metrics in the list from front to back;
      sort_by: [ { "Accuracy": "descend" } ]
```

There are quite a few possible data items in the leaderboard. Not all of them can be shown simultaneously on the screen. In the leaderboard, we provide the ``selected_only`` mode for the user to configure what is shown or is not shown. The user can add his/her interested data items in terms of ``paradigms``, ``modules``, ``hyperparameters``, and ``metrics`` so that the selected columns will be shown.

``` yaml
# benchmarkingJob.yaml
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

Hence, the final `benchmarking.yaml` file will look like:- 

```yaml
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
## Step 4. Execution and Presentation

Finally, the user can run ianvs for benchmarking.

The benchmarking result of the targeted algorithms will be shown after the evaluation is done. Leaderboard examples can be found [here](../leaderboards/leaderboard-in-cloud-edge-collaborative-inference-for-llm/leaderboard-of-cloud-edge-collaborative-inference-for-llm.md).