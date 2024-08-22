import random 
import numpy as np

y = np.random.randint(0,10 ,size=(500,1))
y = np.sort(y, axis=0)
print(y)
class_num = len(np.unique(y))
current_class = random.sample([x for x in range(class_num)], 6)
print(current_class)

indices = np.where((y==current_class))
print(indices)
print(y[indices[0]].shape)
# print(y[497][5])