import tensorflow as tf
import numpy as np


def load_data():
    # load mnist data set
    train_data, valid_data = tf.keras.datasets.mnist.load_data()

    mean = np.mean(train_data[0], axis=0)
    std = np.std(train_data[0], axis=0)

    def normalize(data):
        return (data - mean)/std

    X_train = normalize(train_data[0]).astype(np.float32)
    Y_train = tf.keras.utils.to_categorical(train_data[1]).astype(np.float32)

    X_valid = normalize(valid_data[0]).astype(np.float32)
    Y_valid = tf.keras.utils.to_categorical(valid_data[1]).astype(np.float32)

    return X_train, Y_train, X_valid, Y_valid
