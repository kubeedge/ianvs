import os
import numpy as np
import torch
from sedna.common.class_factory import ClassType, ClassFactory

# set backend
os.environ['BACKEND_TYPE'] = 'PYTORCH'


@ClassFactory.register(ClassType.GENERAL, alias="BaseModel")
class BaseModel:
    def __init__(self, **kwargs):
        self.model = None
    
    def load(self, model_url, **kwargs):
        if model_url:
            print("There is model url==================")
            print(model_url)
            self.model = torch.hub.load('/home/shifan/.cache/torch/hub/ultralytics_yolov5_master', 'custom', path=model_url, source='local')
        else:
            raise Exception("model url does not exist.")
        # self.validator.model = load_my_state_dict(self.validator.model, self.validator.new_state_dict['state_dict'])

    def predict(self, data, **kwargs):
        if type(data) is np.ndarray:
            data = data.tolist()
        with_nms, model_forward_result = kwargs.get("with_nms"), kwargs.get("model_forward_result")
        only_nms, conf = kwargs.get("only_nms"),kwargs.get("conf")
        self.model.eval()
        predictions = []
        if not with_nms:   
            result = self.model(data, with_nms=with_nms, size=640)     
            return result
        else:
            result = self.model(data, model_forward_result=model_forward_result, only_nms=only_nms, conf=conf)            
            predictions.append(np.array(result.pandas().xywhn[0]))
            return predictions

    def evaluate(self, data, **kwargs):
        self.val_args.save_predicted_image = kwargs.get("save_predicted_image", True)
        samples = self._preprocess(data.x)
        predictions = self.predict(samples)
        metric_name, metric_func = kwargs.get("metric")
        if callable(metric_func):
            return metric_func(data.y, predictions)
        else:
            raise Exception(f"not found model metric func(name={metric_name}) in model eval phase")