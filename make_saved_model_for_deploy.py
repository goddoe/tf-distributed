import tensorflow as tf
from utils import save_with_saved_model

model_path = "./output/logs/model.ckpt-480"
graph_path = "{}.meta".format(model_path)

sess = tf.Session()
saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
saver.restore(sess, model_path)


