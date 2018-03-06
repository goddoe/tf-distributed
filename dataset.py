import tensorflow as tf
import numpy as np
import pandas as pd

from utils import print_kwargs


def read_data(data_path, train_ratio, valid_ratio):
    """ Read data and transform data to use in train
    """
    train_df = pd.read_csv(data_path)
    X_train_df = train_df.drop(["target"], axis=1)
    Y_train_df = train_df["target"]

    X_all = X_train_df.as_matrix()
    Y_all = Y_train_df.as_matrix()
    n_class = len(np.unique(Y_all))

    # Y to one hot
    n_sample = len(Y_all)
    tmp = np.zeros((n_sample, n_class))
    tmp[np.arange(n_sample), Y_all] = 1
    Y_all = tmp

    # Shuffle data and split data with train, valid, test sets.
    rand_idx = np.random.permutation(range(len(X_all)))
    X_all = X_all[rand_idx]
    Y_all = Y_all[rand_idx]

    data_num = len(X_all)
    train_data_num = round(data_num * train_ratio)
    valid_data_num = round(data_num * valid_ratio)

    X_train = X_all[:train_data_num]
    Y_train = Y_all[:train_data_num]
    X_valid = X_all[train_data_num:train_data_num + valid_data_num]
    Y_valid = Y_all[train_data_num:train_data_num + valid_data_num]
    X_test = X_all[train_data_num + valid_data_num:]
    Y_test = Y_all[train_data_num + valid_data_num:]

    mean = np.mean(X_train.astype(np.float32), axis=0) + 1e-6
    std = np.std(X_train.astype(np.float32), axis=0) + 1e-6

    def normalize(data):
        return (data.astype(np.float32) + 1e-6 - mean)/std

    X_train = normalize(X_train)
    X_valid = normalize(X_valid)
    X_test = normalize(X_test)

    input_dim = len(X_train[0])
    print_kwargs(input_dim=input_dim,
                 output_dim=n_class,
                 X_train=np.shape(X_train),
                 Y_train=np.shape(Y_train),
                 X_valid=np.shape(X_valid),
                 Y_valid=np.shape(Y_valid),
                 X_test=np.shape(X_test),
                 Y_test=np.shape(Y_test))


    return (X_train.astype(np.float32),
            Y_train.astype(np.float32),
            X_valid.astype(np.float32),
            Y_valid.astype(np.float32),
            X_test.astype(np.float32),
            Y_test.astype(np.float32))


def read_data_2(data_path, train_ratio, valid_ratio):
    # load mnist data set
    train_data, valid_data = tf.keras.datasets.mnist.load_data()

    mean = np.mean(train_data[0].astype(np.float32), axis=0) + 1e-6
    std = np.std(train_data[0].astype(np.float32), axis=0) + 1e-6

    def normalize(data):
        return (data.astype(np.float32) + 1e-6 - mean)/std

    X_train = normalize(train_data[0]).astype(np.float32)
    Y_train = tf.keras.utils.to_categorical(train_data[1]).astype(np.float32)

    X_valid = normalize(valid_data[0]).astype(np.float32)
    Y_valid = tf.keras.utils.to_categorical(valid_data[1]).astype(np.float32)

    return X_train, Y_train, X_valid, Y_valid, None, None


