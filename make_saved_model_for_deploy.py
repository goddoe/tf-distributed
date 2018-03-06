import tensorflow as tf
from utils import save_with_saved_model

# Input
model_path = "./output/logs/model.ckpt-120"
graph_path = "{}.meta".format(model_path)

# Output
export_saved_model_path = "./output/logs/deploy"

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
                      path=export_saved_model_path)

print("*"*30)
print("export_saved_model_path: {}".format(export_saved_model_path))
print("export model done.")
print("*"*30)
