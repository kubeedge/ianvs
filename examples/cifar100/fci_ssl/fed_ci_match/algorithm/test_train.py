import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("./.")
sys.path.append("./..")
from sedna.datasources import TxtDataParse
import numpy as np
import tensorflow as tf
from basemodel import BaseModel
import keras
import logging

logging.getLogger().setLevel(logging.INFO)


def build_classifier(feature_extractor, num_classes, classifier=None):

    if classifier != None:
        new_classifier = keras.Sequential(
            [
                # tf.keras.Input(shape=(None, self.feature_extractor.layers[-2].output_shape[-1])),
                keras.layers.Dense(num_classes, kernel_initializer="lecun_normal")
            ]
        )
        new_classifier.build(
            input_shape=(None, feature_extractor.layers[-2].output_shape[-1])
        )
        new_weights = new_classifier.get_weights()
        old_weights = classifier.get_weights()
        # 复制旧参数
        # weight
        new_weights[0][0 : old_weights[0].shape[0], 0 : old_weights[0].shape[1]] = (
            old_weights[0]
        )
        # bias
        new_weights[1][0 : old_weights[1].shape[0]] = old_weights[1]
        new_classifier.set_weights(new_weights)
        classifier = new_classifier
    else:
        logging.info(f"input shape is {feature_extractor.layers[-2].output_shape[-1]}")
        classifier = keras.Sequential(
            [
                # tf.keras.Input(shape=(None, feature_extractor.layers[-2].output_shape[-1])),
                keras.layers.Dense(num_classes, kernel_initializer="lecun_normal")
            ]
        )
        classifier.build(
            input_shape=(None, feature_extractor.layers[-2].output_shape[-1])
        )

    logging.info(f"finish ! initialize classifier {classifier.summary()}")
    return classifier


def read_data_from_file_to_npy(files, incremental_round=10):
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

    # print(files.x, files.y)
    tasks = []
    for i in range(incremental_round):
        x_train = []
        y_train = []
        start = i * incremental_round
        end = (i + 1) * incremental_round
        print(files.x[start:end])
        for i, file in enumerate(files.x[start:end]):
            x = np.load(file)
            # print(x.shape)
            # print((files.y[][i]))
            y = np.full((x.shape[0],), (files.y[start:end][i]).astype(np.int32))
            x_train.append(x)
            y_train.append(y)
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        tasks.append((x_train, y_train))
        print(x_train.shape, y_train.shape, np.unique(y_train), len(tasks))
    return tasks


def read_data_from_file_to_npy_no_step(files):
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

    # print(files.x, files.y)
    tasks = []
    x_train = []
    y_train = []
    for i, file in enumerate(files.x):
        x = np.load(file)
        # print(x.shape)
        # print((files.y[][i]))
        y = np.full((x.shape[0],), (files.y[i]).astype(np.int32))
        x_train.append(x)
        y_train.append(y)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    tasks.append((x_train, y_train))
    print(x_train.shape, y_train.shape, np.unique(y_train), len(tasks))
    return tasks


def train(
    feature_extractor,
    classifier,
    train_data,
    valid_data=None,
    epochs=60,
    batch_size=128,
    learning_rate=0.01,
    validation_split=0.2,
    num_classes=10,
):
    """Model train"""
    mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
    std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
    all_parameter = []
    all_parameter.extend(feature_extractor.trainable_variables)
    all_parameter.extend(classifier.trainable_variables)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate, weight_decay=0.0001
    )
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=learning_rate, weight_decay=0.0001
    # )
    train_loader = (
        tf.data.Dataset.from_tensor_slices(train_data)
        .shuffle(500000)
        .map(
            lambda x, y: (
                (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std,
                tf.cast(y, dtype=tf.int32),
            )
        )
        .batch(batch_size)
    )
    for epoch in range(epochs):
        epoch_loss = 0
        step = 0
        for _, (x, y) in enumerate(train_loader):
            logging.info(f"step {step} label is {np.unique(y)}")
            with tf.GradientTape() as tape:
                feature = feature_extractor(x)
                y_pred = classifier(feature)
                target = tf.one_hot(y, num_classes)
                loss = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(
                        target, y_pred, from_logits=True
                    )
                )
            pre = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
            y = tf.cast(y, tf.int32)
            acc = tf.reduce_sum(tf.cast(tf.equal(pre, y), tf.float32))
            step += 1
            logging.info(
                f"epoch {epoch} step {step} loss: {loss} acc: {acc} total {x.shape[0]}"
            )
            epoch_loss += loss
            grads = tape.gradient(loss, all_parameter)
            optimizer.apply_gradients(zip(grads, all_parameter))
        logging.info(f"epoch {epoch} loss: {epoch_loss/step}")


def evaluate(feature_extractor, classifier, test_data_x, test_data_y):
    mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
    std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
    test_loader = (
        tf.data.Dataset.from_tensor_slices((test_data_x, test_data_y))
        .map(
            lambda x, y: (
                (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std,
                tf.cast(y, dtype=tf.int32),
            )
        )
        .batch(32)
    )
    acc = 0
    total_num = 0
    total_correct = 0
    for _, (x, y) in enumerate(test_loader):
        # feature = feature_extractor(x)
        # logits = classifier(feature)
        # pred = tf.cast(tf.argmax(logits, axis=1), tf.int64)
        # y = tf.cast(y, tf.int64)
        # acc += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
        # print(x.shape)
        logits = classifier(feature_extractor(x, training=False))
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        pred = tf.reshape(pred, y.shape)
        print(f"pred: {pred} y: {y}")
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        # print(correct)
        correct = tf.reduce_sum(correct, axis=0)

        # print(f"correct: {correct} total: {x.shape[0]}")
        total_num += x.shape[0]
        total_correct += int(correct)

        acc = total_correct / total_num
    logging.info(f"test acc: {acc}")


def task_to_data(task):
    train_data = {}
    train_data["label_x"] = task[0]
    train_data["label_y"] = task[1]
    # print(np.unique(train_data["label_y"]))
    train_data["unlabel_x"] = []
    train_data["unlabel_y"] = []
    return train_data


import tensorflow as tf
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


def build(num_classes: int):
    model = Sequential()
    model.add(
        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
            strides=(2, 2),
            input_shape=(32, 32, 3),
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
    )
    return model


def main_no_incremental_no_split():
    from model import lenet5

    model = lenet5(32, 100)
    train_file = "/home/wyd/ianvs/project/data/cifar100/cifar100_train.txt"
    train_data = TxtDataParse(data_type="train")
    train_data.parse(train_file)
    test_file = "/home/wyd/ianvs/project/data/cifar100/cifar100_test.txt"
    test_data = TxtDataParse(data_type="eval")
    test_data.parse(test_file)
    # print(train_data.x, train_data.y)
    # print(test_data.x, test_data.y)
    incremental_round = 1
    test_task = read_data_from_file_to_npy_no_step(test_data)[0]
    print(test_task[0].shape, test_task[1].shape)
    tasks = read_data_from_file_to_npy_no_step(train_data)
    train_data = tasks[0]
    mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
    std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, weight_decay=0.0001)
    # optimizer = tf.keras.optimizers.Adam(
    #     learning_rate=learning_rate, weight_decay=0.0001
    # )
    train_loader = (
        tf.data.Dataset.from_tensor_slices(train_data)
        .shuffle(500000)
        .map(
            lambda x, y: (
                (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std,
                tf.cast(y, dtype=tf.int32),
            )
        )
        .batch(32)
    )
    for epoch in range(200):
        epoch_loss = 0
        step = 0
        total_correct = 0
        total_num = 0
        for _, (x, y) in enumerate(train_loader):
            logging.info(f"step {step} label is {np.unique(y)}")
            with tf.GradientTape() as tape:
                y_pred = model(x)
                target = tf.one_hot(y, 100)
                loss = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(
                        target, y_pred, from_logits=True
                    )
                )
            pre = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
            y = tf.cast(y, tf.int32)
            acc = tf.reduce_sum(tf.cast(tf.equal(pre, y), tf.float32))
            step += 1
            total_correct += int(acc)
            total_num += x.shape[0]
            epoch_loss += loss
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        logging.info(
            f"epoch {epoch} loss: {epoch_loss/step} acc : {total_correct/total_num}"
        )


def main_no_incremental():
    train_file = "/home/wyd/ianvs/project/data/cifar100/cifar100_train.txt"
    train_data = TxtDataParse(data_type="train")
    train_data.parse(train_file)
    test_file = "/home/wyd/ianvs/project/data/cifar100/cifar100_test.txt"
    test_data = TxtDataParse(data_type="eval")
    test_data.parse(test_file)
    # print(train_data.x, train_data.y)
    # print(test_data.x, test_data.y)
    incremental_round = 1
    test_task = read_data_from_file_to_npy_no_step(test_data)[0]
    print(test_task[0].shape, test_task[1].shape)
    tasks = read_data_from_file_to_npy_no_step(train_data)
    config = {
        "learning_rate": 0.01,
        "epochs": 100,
        "batch_size": 128,
        "task_size": 100,
        "memory_size": 2000,
    }
    estimator = BaseModel(**config)
    feature_extractor = estimator.FedCiMatch.feature_extractor
    classifier = keras.Sequential(
        [
            # tf.keras.Input(shape=(None, self.feature_extractor.layers[-2].output_shape[-1])),
            keras.layers.Dense(100, kernel_initializer="lecun_normal")
        ]
    )
    classifier.build(input_shape=(None, feature_extractor.layers[-2].output_shape[-1]))
    train_data = tasks[0]
    train(feature_extractor, classifier, train_data, epochs=60)
    feature_extractor.save_weights("./feature_extractor.weights.h5")
    evaluate(feature_extractor, classifier, test_task[0], test_task[1])


def load_data_from_tf(incremental_round=10):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    class_labels = np.unique(y_train)
    train_data_dict = {label: [] for label in class_labels}
    train_label_dict = {label: [] for label in class_labels}
    for img, label in zip(x_train, y_train):
        train_data_dict[label[0]].append(img)
        train_label_dict[label[0]].append(label)
    tasks = []
    for i in range(incremental_round):
        x_data = []
        y_data = []
        start = i * incremental_round
        end = (i + 1) * incremental_round
        for label in range(start, end):
            x_data.append(np.array(train_data_dict[label]))
            y_data.append(np.array(train_label_dict[label]))
        x_data = np.concatenate(x_data, axis=0)
        y_data = np.array(y_data).reshape(-1)
        tasks.append((x_data, y_data))
        print(x_data.shape, y_data.shape, np.unique(y_data), len(tasks))
    return tasks


def main_incremental(incremental_round=10):
    # train_file = "/home/wyd/ianvs/project/data/cifar100/cifar100_train.txt"
    # train_data = TxtDataParse(data_type="train")
    # train_data.parse(train_file)
    test_file = "/home/wyd/ianvs/project/data/cifar100/cifar100_test.txt"
    test_data = TxtDataParse(data_type="eval")
    test_data.parse(test_file)
    # train_task = read_data_from_file_to_npy(train_data, incremental_round)
    test_task = read_data_from_file_to_npy(test_data, incremental_round)
    print("____________--------------__________________")
    train_task = load_data_from_tf(incremental_round)

    config = {
        "learning_rate": 0.01,
        "epochs": 20,
        "batch_size": 128,
        "task_size": 10,
        "memory_size": 2000,
    }
    estimator = BaseModel(**config)
    # feature_extractor = estimator.FedCiMatch.feature_extractor
    # classifier = None

    for i in range(incremental_round):
        train_data = task_to_data(train_task[i])
        estimator.train(train_data, val_data=None, task_id=i, round=1)
        feature_extractor = estimator.FedCiMatch.feature_extractor
        classifier = estimator.FedCiMatch.classifier
        evaluate(feature_extractor, classifier, test_task[i][0], test_task[i][1])
    final_test_data = read_data_from_file_to_npy_no_step(test_data)[0]
    evaluate(feature_extractor, classifier, final_test_data[0], final_test_data[1])


if __name__ == "__main__":
    main_incremental()
