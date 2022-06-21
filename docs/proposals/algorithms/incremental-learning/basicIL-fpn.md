# Incremental learning: BasicIL-FPN

Here we will show how to implement a single task learning algorithm for testing in ianvs, based on an opensource algorithm [FPN].

For test of your own algorithm, FPN is not required. It can be replaced to any algorithm, as long as it complies the requirement of ianvs interface.


Before start, it should be known that Ianvs testing algorithm development depends on Sedna Lib. The following is recommended development workflow:
1. Develop by yourself: put the algorithm implementation to ianvs [examples directory] locally, for testing.
2. Publish to everyone: submit the algorithm implementation to [Sedna repository], for sharing, then everyone can test and use your algorithm.

## Customize algorithm

Sedna provides a class called `class_factory.py` in `common` package, in which only a few lines of changes are required to become a module of sedna.

Two classes are defined in `class_factory.py`, namely `ClassType` and `ClassFactory`.

`ClassFactory` can register the modules you want to reuse through decorators. For example, in the following code example, you have customized an **single task learning algorithm**, you only need to add a line of `ClassFactory.register(ClassType.GENERAL)` to complete the registration.

The following code is just to show the overall structure of a basicIL-fpn BaseModel, not the complete version. The complete code can be found [here](https://github.com/JimmyYang20/ianvs/tree/main/examples/pcb-aoi/incremental_learning_bench/testalgorithms/fpn).

```python
@ClassFactory.register(ClassType.GENERAL, "estimator")
class BaseModel:
    def __init__(self, **kwargs):
        """
        initialize logging configuration
        """
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.session = tf.Session(
            config=sess_config, graph=self.graph)

        self.restorer = None
        self.checkpoint_path = self.load(Context.get_parameters("base_model_url"))
        self.temp_dir = tempfile.mkdtemp()
        if not os.path.isdir(self.temp_dir):
            mkdir(self.temp_dir)

        os.environ["MODEL_NAME"] = "model.zip"
        cfgs.LR = kwargs.get("learning_rate", 0.0001)
        cfgs.MOMENTUM = kwargs.get("momentum", 0.9)
        cfgs.MAX_ITERATION = kwargs.get("max_iteration", 5)

    def train(self, train_data, valid_data=None, **kwargs):
        """
        train
        """

        if train_data is None or train_data.x is None or train_data.y is None:
            raise Exception("Train data is None.")

        with tf.Graph().as_default():

            img_name_batch, train_data, gtboxes_and_label_batch, num_objects_batch, data_num =
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

        inference_output_dir = os.getenv("INFERENCE_OUTPUT_DIR")

        with tf.Graph().as_default():

            img_plac = tf.placeholder(shape=[None, None, 3], dtype=tf.uint8)

            img_tensor = tf.cast(img_plac, tf.float32) - tf.constant([103.939, 116.779, 123.68])
            img_batch = image_preprocess.short_side_resize_for_inference_data(img_tensor,
                                                                              target_shortside_len=cfgs.SHORT_SIDE_LEN,
                                                                              is_resize=True)


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

            print(f"load {model_url} to {self.checkpoint_path}")
        else:
            raise Exception(f"model url is None")

        return self.checkpoint_path

    def test(self, valid_data, **kwargs):
        '''
        output the test results and groudtruth
        while this function is not in sedna's incremental learning interfaces
        '''

        checkpoint_path = kwargs.get("checkpoint_path")
        img_name_batch = kwargs.get("img_name_batch")
        gtboxes_and_label_batch = kwargs.get("gtboxes_and_label_batch")
        num_objects_batch = kwargs.get("num_objects_batch")
        graph = kwargs.get("graph")
        data_num = kwargs.get("data_num")

        test.fpn_test(validate_data=valid_data,
                      checkpoint_path=checkpoint_path,
                      graph=graph,
                      img_name_batch=img_name_batch,
                      gtboxes_and_label_batch=gtboxes_and_label_batch,
                      num_objects_batch=num_objects_batch,
                      data_num=data_num)

    def evaluate(self, data, model_path, **kwargs):
        if data is None or data.x is None or data.y is None:
            raise Exception("Prediction data is None")

        self.load(model_path)
        predict_dict = self.predict(data.x)

        metric = kwargs.get("metric")
        if callable(metric):
            return {"f1_score": metric(data.y, predict_dict)}
        return {"f1_score": f1_score(data.y, predict_dict)}
```

After registration, you only need to change the name of the basicIL and parameters in the yaml file, and then the corresponding class will be automatically called according to the name.



[FPN]: https://github.com/DetectionTeamUCAS/FPN_Tensorflow
[examples directory]: ../../../../examples
[Sedna repository]: https://github.com/kubeedge/sedna

