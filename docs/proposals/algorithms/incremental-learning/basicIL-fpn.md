# Incremental learning: BasicIL-FPN

Here we will show how to implement a single task learning algorithm for testing in ianvs, based on an opensource algorithm [FPN].

For test of your own algorithm, FPN is not required. It can be replaced to any algorithm, as long as it complies the requirement of ianvs interface.


## Customize algorithm

Sedna provides a class called `class_factory.py` in `common` package, in which only a few lines of changes are required to become a module of sedna.

Two classes are defined in `class_factory.py`, namely `ClassType` and `ClassFactory`.

`ClassFactory` can register the modules you want to reuse through decorators. For example, in the following code example, you have customized an **single task learning algorithm**, you only need to add a line of `ClassFactory.register(ClassType.GENERAL)` to complete the registration.

```python

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

After registration, you only need to change the name of the basicIL and parameters in the yaml file, and then the corresponding class will be automatically called according to the name.



[FPN]: https://github.com/DetectionTeamUCAS/FPN_Tensorflow
>>>>>>> fc0bc95... Add FPN & BasicIL algorithm
