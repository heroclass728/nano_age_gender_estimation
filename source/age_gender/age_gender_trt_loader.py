import tensorflow as tf
import numpy as np

from keras.preprocessing import image
from settings import TRT_MODEL_INPUT_NAMES, TRT_MODEL_OUTPUT_NAMES, AGE_TRT_MODEL_PATH, GENDER_TRT_MODEL_PATH


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


class TRTPredictor:

    def __init__(self, trt_pb_path, sess_name):

        trt_graph = get_frozen_graph(trt_pb_path)

        # Create session and load graph
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.tf_sess = tf.Session(config=tf_config)
        tf.import_graph_def(trt_graph, name=sess_name)

        # Get graph input size
        self.image_size = []
        for node in trt_graph.node:
            if '_input' in node.name:
                size = node.attr['shape'].shape
                self.image_size = [size.dim[i].size for i in range(1, 4)]
                break
        # print("image_size: {}".format(self.image_size))

        # input and output tensor names.
        self.input_tensor_name = sess_name + "/%s:0" % TRT_MODEL_INPUT_NAMES[0]
        output_tensor_name = sess_name + "/%s:0" % TRT_MODEL_OUTPUT_NAMES[0]

        # print("input_tensor_name: {}\noutput_tensor_name: {}".format(self.input_tensor_name, output_tensor_name))

        self.output_tensor = self.tf_sess.graph.get_tensor_by_name(output_tensor_name)

    def predict(self, frame):

        feed_dict = {
            self.input_tensor_name: frame
        }
        preds = self.tf_sess.run(self.output_tensor, feed_dict)

        return preds


if __name__ == '__main__':

    img_path = '/media/mensa/Data/Task/JetsonAgeGender/man.jpeg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255

    gender_model = TRTPredictor(trt_pb_path=GENDER_TRT_MODEL_PATH, sess_name='gender')
    age_model = TRTPredictor(trt_pb_path=AGE_TRT_MODEL_PATH, sess_name="age")
    age = age_model.predict(frame=x)
    gender = gender_model.predict(frame=x)

    print("")
