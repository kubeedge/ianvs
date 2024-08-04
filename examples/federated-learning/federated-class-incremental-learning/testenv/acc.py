import tensorflow as tf
import numpy as np
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ['acc']


@ClassFactory.register(ClassType.GENERAL, alias='accuracy')
def accuracy(y_true, y_pred, **kwargs):
    # print(f"y_pred: {y_pred}")
    y_pred_arr = [val for val in y_pred.values()]
    y_true_arr = []
    for i in range(len(y_pred_arr)):
        y_true_arr.append(np.full(y_pred_arr[i].shape, int(y_true[i])))
    y_pred = tf.cast(tf.convert_to_tensor(np.concatenate(y_pred_arr, axis=0)), tf.int64)

    y_true = tf.cast(tf.convert_to_tensor(np.concatenate(y_true_arr, axis=0)), tf.int64)
    # print(tf.shape(y_true), tf.shape(y_pred))
    total = tf.shape(y_true)[0]
    correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.int32))
    acc = float(int(correct) / total)
    print(f"acc:{acc}")
    return acc

