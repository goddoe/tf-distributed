import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def mlp(X,
        output_dim,
        is_training,
        fc_channel,
        reg_lambda,
        dropout_keep_prob,
        name="mlp"):
    """Build Model
    User can modify codes below
    """

    with tf.variable_scope(name):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.random_normal_initializer(),
                            weights_regularizer=slim.l2_regularizer(reg_lambda)):
            h = slim.flatten(X)
            print("="*30)
            print(fc_channel[0])
            print("="*30)
            h = slim.fully_connected(h, fc_channel[0], scope='fc/fc_1')
            h = slim.dropout(h, dropout_keep_prob,
                             is_training=is_training, scope='fc/fc_1/dropout')
            h = slim.fully_connected(h, fc_channel[1], scope='fc/fc_2')
            h = slim.dropout(h, dropout_keep_prob,
                             is_training=is_training, scope='fc/fc_2/dropout')
            logits = slim.fully_connected(
                h, output_dim, activation_fn=None, scope='logits')

    return logits


model_f = mlp
