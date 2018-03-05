import os
import json

import tensorflow as tf


# python utils

def load_json(path):
    with open(path, "rt") as f:
        return json.load(f)


def makedirs(path):
    try:
        os.makedirs(path)
    except Exception as e:
        return False
    return True


def print_kwargs(**kwargs):
    for key, val in kwargs.items():
        print("{}: {}".format(key, val))


# tf utils

def calc_metric(Y, Y_pred, name=None):
    with tf.variable_scope(name or 'metric'):
        correct_prediction = tf.equal(
            tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    return accuracy, correct_prediction


def save_with_saved_model(sess, X, Y_pred, path):
        """Description

        Args:
            path: save path
        """
        if not os.path.exists(os.path.dirname(path)):
            makedirs(os.path.dirname(path))
        try:
            builder = tf.saved_model.builder.SavedModelBuilder(path)
            tensor_info_X = tf.saved_model.utils.build_tensor_info(X)
            tensor_info_Y = tf.saved_model.utils.build_tensor_info(Y_pred)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'X': tensor_info_X},
                    outputs={'Y_pred': tensor_info_Y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            init = tf.tables_initializer()
            legacy_init_op = tf.group(
                init, name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={'predict': prediction_signature},
                legacy_init_op=legacy_init_op)
            builder.save()
        except Exception as e:
            raise Exception("Error in save_with_saved_model: {}".format(str(e)))

        return sess
