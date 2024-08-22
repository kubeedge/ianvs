import keras
import tensorflow as tf
import numpy as np
from keras.src.layers import Dense
from keras.src.models.cloning import clone_model
from resnet  import resnet10


class NetWork(keras.Model):
    def __init__(self, num_classes, feature_extractor):
        super(NetWork, self).__init__()
        self.feature = feature_extractor
        self.fc = Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.feature(inputs)
        x = self.fc(x)
        return x


    def feature_extractor(self, inputs):
        return self.feature(inputs)

    def predict(self, fea_input):
        return self.fc(fea_input)
    


def incremental_learning(old_model:NetWork, num_class):
    new_model = NetWork(num_class, resnet10(num_class) )
    x = np.random.rand(1, 32, 32, 3)
    y = np.random.randint(0, num_class, 1)
    new_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    new_model.fit(x, y, epochs=1)
    print(old_model.fc.units, new_model.fc.units)
    for layer in old_model.layers:
        if hasattr(new_model.feature, layer.name):
            new_model.feature.__setattr__(layer.name, layer)
    if num_class > old_model.fc.units:

        original_use_bias = hasattr(old_model.fc, 'bias')
        print("original_use_bias", original_use_bias)
        init_class = old_model.fc.units
        k = new_model.fc.kernel
        new_model.fc.kernel.assign(tf.pad(old_model.fc.kernel, 
        [[0, 0], [0, num_class - init_class]]))  # 假设初始类别数为10
        if original_use_bias:
            new_model.fc.bias.assign(tf.pad(old_model.fc.bias, 
            [[0, num_class - init_class]]))
   

    new_model.build((None, 32, 32, 3))
    return new_model
