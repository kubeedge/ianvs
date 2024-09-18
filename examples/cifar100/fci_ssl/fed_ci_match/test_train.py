from sedna.datasources import TxtDataParse
import numpy as np
import tensorflow as tf
from algorithm.model import resnet10
import keras
import logging
logging.getLogger().setLevel(logging.INFO)
def build_classifier(feature_extractor):
    classifier = keras.Sequential([
        # tf.keras.Input(shape=(None, self.feature_extractor.layers[-2].output_shape[-1])),
        keras.layers.Dense(10, kernel_initializer='lecun_normal')
    ])
    classifier.build(input_shape=(None, feature_extractor.layers[-2].output_shape[-1]))
    logging.info(f"finish ! initialize classifier {classifier.summary()}")
    return classifier
def read_data_from_file_to_npy( files):
    """
    read data from file to numpy array

    Parameters
    ---------
    files: list
        the address url of data file.

    Returns
    -------
    list
        data in numpy array.

    """
    x_train = []
    y_train = []
    print(files.x, files.y)
    for i, file in enumerate(files.x[:10]):
        x = np.load(file)
        # print(x.shape)
        # print(type(files.y[i]))
        y = np.full((x.shape[0], 1), (files.y[i]).astype(np.int32))
        x_train.append(x)
        y_train.append(y)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    print(x_train.shape, y_train.shape)
    return x_train, y_train

def train(feature_extractor, classifier,
              train_data, valid_data=None,
              epochs=60,
              batch_size=128,
              learning_rate=0.01,
              validation_split=0.2):
        """ Model train """
        mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        all_parameter = []
        all_parameter.extend(feature_extractor.trainable_variables)
        all_parameter.extend(classifier.trainable_variables)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        train_loader = tf.data.Dataset.from_tensor_slices(train_data).shuffle(500000).map(
            lambda x,y:(
                (tf.cast(x, dtype=tf.float32) / 255. - mean) / std,
                tf.cast(y, dtype=tf.int32)
            )
        ).batch(batch_size)
        for epoch in range(epochs):
            epoch_loss = 0
            step = 0
            for _, (x, y) in enumerate(train_loader):
                with tf.GradientTape() as tape:
                    feature = feature_extractor(x)
                    logits = tf.nn.softmax(classifier(feature))
                    loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(y , logits))
                step += 1
                logging.info(f"epoch {epoch} step {step} loss: {loss}")
                epoch_loss += loss
                grads = tape.gradient(loss, all_parameter)
                optimizer.apply_gradients(zip(grads, all_parameter))
            logging.info(f"epoch {epoch} loss: {epoch_loss/step}")

def evaluate(feature_extractor, classifier, test_data_x, test_data_y):
    mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
    std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
    test_loader = tf.data.Dataset.from_tensor_slices((test_data_x, test_data_y)).map(
        lambda x,y:(
            (tf.cast(x, dtype=tf.float32) / 255. - mean) / std,
            tf.cast(y, dtype=tf.int32)
        )
    ).batch(32)
    acc = 0
    total_num = 0
    total_correct = 0
    for _, (x, y) in enumerate(test_loader):
        # feature = feature_extractor(x)
        # logits = classifier(feature)
        # pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)
        # y = tf.cast(y, tf.int64)
        # acc += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
        logits = classifier(feature_extractor(x, training=False))
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        pred = tf.reshape(pred, y.shape)
        # print(f"pred: {pred} y: {y}")
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        # print(correct)
        correct = tf.reduce_sum(correct, axis=0)
        
        # print(f"correct: {correct} total: {x.shape[0]}")
        total_num += x.shape[0]
        total_correct += int(correct)

        acc = total_correct / total_num
    logging.info(f"test acc: {acc}")

def main():
    train_file = '/home/wyd/ianvs/project/data/cifar100/cifar100_train.txt'
    train_data = TxtDataParse(data_type='train')
    train_data.parse(train_file)
    train_data = read_data_from_file_to_npy(train_data)
    train_data_x = train_data[0]
    train_data_y = train_data[1]
    print(len(train_data_x), len(train_data_y))
    test_file = '/home/wyd/ianvs/project/data/cifar100/cifar100_test.txt'
    test_data = TxtDataParse(data_type='eval')
    test_data.parse(test_file)
    test_data = read_data_from_file_to_npy(test_data)
    test_data_x = test_data[0]
    test_data_y = test_data[1]
    feature_extractor = resnet10()
    feature_extractor.build((None, 32,32,3))
    feature_extractor.call(keras.Input(shape=(32,32,3)))
    classifier = build_classifier(feature_extractor)
    train(feature_extractor, classifier, train_data)
    evaluate(feature_extractor, classifier, test_data_x, test_data_y)
    
import time
import threading 
def local_thread(t):
    time.sleep(3)
    return t

def local_thread_v2(t):
    val = local_thread(t)
    print(val, t)
if __name__ == '__main__':
    a = []
    for i in range(3):
        t = threading.Thread(target=local_thread_v2, args=(i,))
        t.start()
        a.append(t)
    for t in a:
        t.join()