import tensorflow as tf
import keras
from algorithm.network import NetWork, incremental_learning
from algorithm.model import resnet10, lenet5
from sedna.datasources import TxtDataParse
from core.testenvmanager.dataset.utils import read_data_from_file_to_npy
import copy 
train_file = '/home/wyd/ianvs/project/data/cifar100/cifar100_train.txt'
train_data = TxtDataParse(data_type='train')
train_data.parse(train_file)
train_data = read_data_from_file_to_npy(train_data)
train_loader = tf.data.Dataset.from_tensor_slices(train_data).shuffle(500000).batch(32)
x_train, y_train = train_data
task_id = 0
fe = resnet10(10)
model = NetWork(100, fe)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
for range in range(100):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    x_task_data, y_task_data = x_train[task_id*5000 : (task_id+1)*5000], y_train[task_id*5000 : (task_id+1)*5000]
    task_loader = tf.data.Dataset.from_tensor_slices((x_task_data, y_task_data)).shuffle(5000).batch(128)
    # if range != 0 and range % 10 == 0:
    #     task_id += 1
    #     model = incremental_learning(model, 10*(task_id+1))
    for x, y in task_loader:
        model.fit(x, y, epochs=1)
        # print(y)
        # with tf.GradientTape() as tape:
        #     logits = model(x, training=True)
        #     label =y
        #     # print(y.shape[0])
        #     y = tf.one_hot(y, 100)
        #     y = tf.squeeze(y, axis=1)
        #     loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, logits, from_logits=True))
        #     pred = tf.argmax(logits, axis=1)
        #     pred = tf.cast(pred, dtype=tf.int32)
        #     pred = tf.reshape(pred, label.shape)
        #     # print(pred.shape, label.shape)
        #     correct = tf.cast(tf.equal(pred, label), dtype=tf.int32)
        #     # print(correct.shape)
        #     correct = tf.reduce_sum(correct)
        #     # print(correct, y.shape[0])
        # grads = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply(grads, model.trainable_variables)
        # print(f'loss: {loss}, accuracy: {correct / x.shape[0]}')
