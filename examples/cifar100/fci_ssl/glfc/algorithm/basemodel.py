import os 
import numpy as np
import keras
import tensorflow as tf
from sedna.common.class_factory import ClassType, ClassFactory
from model import resnet10, lenet5
from network import NetWork, incremental_learning
from GLFC import GLFC_Client
import logging 

os.environ['BACKEND_TYPE'] = 'KERAS'
__all__ = ["BaseModel"]
logging.getLogger().setLevel(logging.INFO)

@ClassFactory.register(ClassType.GENERAL, alias='glfc')
class BaseModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.epochs = kwargs.get('epochs', 1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.task_size = kwargs.get('task_size', 10)
        self.memory_size = kwargs.get('memory_size', 2000)
        self.encode_model = lenet5(32, 100)
        # self.fe = self.build_feature_extractor()
        self.num_classes = 10 # the number of class for the first task
        self.GLFC_Client = GLFC_Client( self.num_classes, self.batch_size, self.task_size, self.memory_size, self.epochs, self.learning_rate, self.encode_model)
        self.best_old_model = [] 
        self.class_learned = 0
        self.fe_weights_length = len(self.GLFC_Client.feature_extractor.get_weights())

    def get_weights(self):
        print("get weights")
        weights = []
        fe_weights = self.GLFC_Client.feature_extractor.get_weights()
        clf_weights = self.GLFC_Client.classifier.get_weights()
        weights.extend(fe_weights)
        weights.extend(clf_weights)
        return weights
    
    def set_weights(self, weights):
        print("set weights")
        fe_weights = weights[:self.fe_weights_length]
       
        clf_weights = weights[self.fe_weights_length:]
        self.GLFC_Client.feature_extractor.set_weights(fe_weights)
        self.GLFC_Client.classifier.set_weights(clf_weights)
        
    def train(self, train_data,val_data, **kwargs):
        task_id = kwargs.get('task_id', 0)
        round = kwargs.get('round', 1)
        logging.info(f"in train: {round} task_id:  {task_id}")
        self.class_learned += self.task_size
        self.GLFC_Client.before_train(task_id, train_data, self.class_learned,  old_model=self.best_old_model)
        
        self.GLFC_Client.train(round)
        proto_grad = self.GLFC_Client.proto_grad()
        print(type(proto_grad))
        # self.GLFC_Client.evaluate()
        return {'num_samples': len(train_data[0]) , 'proto_grad' : proto_grad, 'task_id': task_id}
    
    def helper_function(self, helper_info, **kwargs):
        self.best_old_model = helper_info['best_old_model']
        
    
    def predict(self, data, **kwargs):
        result = {}
        for data in data.x:
            x = np.load(data)
            logits = self.GLFC_Client.model_call(x,training=False)
            pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            result[data] = pred.numpy()
        print("finish predict")
        return result