import argparse
import sys

import tensorflow as tf

slim = tf.contrib.slim

FLAGS = None
INPUT_DIM = 784
OUTPUT_DIM = 10


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

            h = slim.fully_connected(X, fc_channel, scope='fc/fc_1')
            h = slim.dropout(h, dropout_keep_prob, is_training=is_training, scope='fc/dropout')
            logits = slim.fully_connected(h, output_dim, activation_fn=None, scope='fc/fc_2')

    return logits


def calc_metric(Y, Y_pred, name=None):
    with tf.variable_scope(name or 'metric'):
        correct_prediction = tf.equal(
            tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    return accuracy, correct_prediction


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            X = tf.placeholder(dtype=tf.float32,
                               shape=[None, INPUT_DIM],
                               name="X")
            Y = tf.placeholder(dtype=tf.float32,
                               shape=[None, OUTPUT_DIM],
                               name="Y")

            is_training = tf.placeholder_with_default(False,
                                                      shape=None,
                                                      name="is_training")
                               
            logits = mlp(X=X,
                         output_dim=OUTPUT_DIM,
                         is_training=is_training,
                         fc_channel=128,
                         reg_lambda=0.,
                         dropout_keep_prob=0.8)

            Y_pred = slim.softmax(logits)

            loss = slim.losses.softmax_cross_entropy(logits, Y)
            accuracy, correct = calc_metric(Y, Y_pred)

            global_step = tf.contrib.framework.get_or_create_global_step()

            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                loss, global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(
                                                   FLAGS.task_index == 0),
                                               checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run(train_op)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
