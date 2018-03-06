import argparse

import tensorflow as tf
from utils import save_with_saved_model

parser = argparse.ArgumentParser(description='Make deploy model.')
parser.add_argument(
    "--model_path",
    type=str,
    default="",
    help="path of model")

parser.add_argument(
    "--deploy_model_path",
    type=str,
    default="",
    help="path of saved model")

p = parser.parse_args()


# Input
model_path = p.model_path
graph_path = "{}.meta".format(model_path)

# Output
deploy_model_path = p.deploy_model_path

# Main
print("="*30)
print("Model Load Start")

sess = tf.Session()
saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
saver.restore(sess, model_path)

X = tf.get_collection('X')[0]
Y_pred = tf.get_collection('Y_pred')[0]

save_with_saved_model(sess=sess,
                      X=X,
                      Y_pred=Y_pred,
                      path=deploy_model_path)

print("*"*30)
print("deploy_model_path: {}".format(deploy_model_path))
print("export model done.")
print("*"*30)
