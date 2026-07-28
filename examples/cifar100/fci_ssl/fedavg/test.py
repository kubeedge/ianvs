from algorithm.resnet import resnet10
from algorithm.network import NetWork, incremental_learning
import copy 
import numpy as np
fe = resnet10(10)
model = NetWork(10, fe)
new_model = copy.deepcopy(model)

x = np.random.rand(1, 32, 32, 3)
y = np.random.randint(0, 10, 1)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=1)
new_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
new_model.fit(x, y, epochs=1)