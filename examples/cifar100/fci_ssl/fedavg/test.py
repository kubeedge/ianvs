<<<<<<< HEAD
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
=======
version https://git-lfs.github.com/spec/v1
oid sha256:4939a3bb75a37464969fdfe787ed005d7ff2d4d514a495bcf801df7a73d958e7
size 516
>>>>>>> 9676c3e (ya toh aar ya toh par)
