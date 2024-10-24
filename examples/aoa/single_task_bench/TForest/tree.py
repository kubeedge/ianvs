from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error

class Tree:
    def __init__(self, train_features, test_features, train_labels, test_labels, model=AdaBoostRegressor(random_state=42)):
        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.model = model
        self.r2 = None
        self.checked = False

    def train(self):
        self.model.fit(self.train_features, self.train_labels)
        self.r2 = -mean_absolute_error(self.model.predict(self.test_features), self.test_labels)

    def predict(self, features):
        return self.model.predict(features)

    def predict_score(self, model):
        return -mean_absolute_error(model.predict(self.test_features), self.test_labels)
