import json

import tensorflow as tf


def calc_metric(Y, Y_pred, name=None):
    with tf.variable_scope(name or 'metric'):
        correct_prediction = tf.equal(
            tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    return accuracy, correct_prediction


def load_json(path):
    with open(path, "rt") as f:
        return json.load(f)


