import argparse
import sys

import tensorflow as tf

from model import mlp
from dataset import load_data
from utils import calc_metric, load_json

slim = tf.contrib.slim

FLAGS = None

VERBOSE_INTERVAL = 10  # by batch
TRAIN_ACC_WINDOW_SIZE = 10


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    # Parameters 
    param_dict = load_json(FLAGS.param_path)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # Load Data
        (X_train,
         Y_train,
         X_valid,
         Y_valid) = load_data()
            
        # Inference output dimension
        output_dim = len(Y_train[0]) 

        # check chief
        is_chief = FLAGS.task_index == 0

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model...
            # Datasets
            train_X_dataset = tf.data.Dataset.from_tensor_slices(X_train)
            train_Y_dataset = tf.data.Dataset.from_tensor_slices(Y_train)
            train_dataset = tf.data.Dataset.zip((train_X_dataset, train_Y_dataset))
            train_dataset = train_dataset.shuffle(1000).batch(param_dict['batch_size']).repeat(param_dict['n_epoch'])

            if is_chief:
                valid_X_dataset = tf.data.Dataset.from_tensor_slices(X_valid)
                valid_Y_dataset = tf.data.Dataset.from_tensor_slices(Y_valid)
                valid_dataset = tf.data.Dataset.zip((valid_X_dataset, valid_Y_dataset))
                valid_dataset = valid_dataset.shuffle(1000).batch(param_dict['batch_size'])

            # Feedable Iterator
            handle = tf.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
                handle, train_dataset.output_types, train_dataset.output_shapes)

            # Iterators 
            train_iterator = train_dataset.make_one_shot_iterator()
            train_handle_tensor = train_iterator.string_handle()

            if is_chief:
                valid_iterator = valid_dataset.make_initializable_iterator()
                valid_handle_tensor = valid_iterator.string_handle()

            X, Y = iterator.get_next()
            is_training = tf.placeholder_with_default(False,
                                                      shape=None,
                                                      name="is_training")

            global_step = tf.contrib.framework.get_or_create_global_step()
                               
            logits = mlp(X=X,
                         output_dim=output_dim,
                         is_training=is_training,
                         **param_dict['model_param'])

            Y_pred = slim.softmax(logits)

            loss = slim.losses.softmax_cross_entropy(logits, Y)
            accuracy, correct = calc_metric(Y, Y_pred)

            train_op = tf.train.AdamOptimizer(0.01).minimize(
                loss, global_step=global_step)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks) as mon_sess:

            # Get dataset handle
            train_handle = mon_sess.run(train_handle_tensor)
            valid_handle = mon_sess.run(valid_handle_tensor)

            acc_window = [0.]*TRAIN_ACC_WINDOW_SIZE
            
            batch_i = 0
            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run(train_op,
                             feed_dict={is_training: True,
                                        handle: train_handle, })
                if is_chief:
                    train_accuracy = mon_sess.run(accuracy,
                                                  feed_dict={is_training: False,
                                                             handle: train_handle, })
                    acc_window = acc_window[1:] + [train_accuracy]

                    if batch_i % VERBOSE_INTERVAL == 0:
                        recent_mean_train_accuracy = sum(acc_window)/len(acc_window)

                        valid_i = 0 
                        mon_sess.run(valid_iterator.initializer)
                        valid_correct = 0 
                        valid_num = 0

                        while True:
                            try:
                                pred, batch_valid_correct = mon_sess.run([Y_pred, correct],
                                                                         feed_dict={is_training: False,
                                                                                    handle: valid_handle, })
                                valid_correct += batch_valid_correct.sum()
                                valid_num += pred.shape[0]
                                valid_i += 1
                            except tf.errors.OutOfRangeError:
                                break
                        valid_accuracy = valid_correct/valid_num

                        print("recent_mean_train_accuracy : {}".format(recent_mean_train_accuracy))
                        print("valid_accuracy : {}".format(valid_accuracy))

                batch_i += 1


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
    parser.add_argument(
        "--param_path",
        type=str,
        default="./param.json",
        help="parameters of model"
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
