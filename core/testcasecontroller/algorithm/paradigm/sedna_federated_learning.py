from sedna.core.base import JobBase

class FederatedLearning(JobBase):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def train(self, train_data, vald_data, **kwargs):
        return self.estimator.train(train_data, vald_data, **kwargs)
    
    def get_weights(self):
        return self.estimator.get_weights()
    
    def set_weights(self, weights):
        self.estimator.set_weights(weights)
    
    def helper_function(self, helper_info):
        return self.estimator.helper_function(helper_info)