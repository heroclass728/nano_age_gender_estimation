import os
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

from tensorflow.python.framework import graph_io
from keras.models import load_model
from settings import CUR_DIR


def freeze_graph(graph, session, output, save_pb_dir, save_pb_name, save_pb_as_text=False):

    with graph.as_default():
        graph_def_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graph_def_frozen = tf.graph_util.convert_variables_to_constants(session, graph_def_inf, output)
        graph_io.write_graph(graph_def_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)

        return graph_def_frozen


def convert_h5_to_pb(model_p, save_pb_dir, pb_name):
    # This line must be executed before loading Keras model.
    tf.keras.backend.set_learning_phase(0)
    model = load_model(model_p)

    session = tf.keras.backend.get_session()

    input_names = [t.op.name for t in model.inputs]
    output_names = [t.op.name for t in model.outputs]

    # Prints input and output nodes names, take notes of them.
    print(input_names, output_names)

    frozen_graph = freeze_graph(graph=session.graph, session=session, output=output_names, save_pb_dir=save_pb_dir,
                                save_pb_name=pb_name)

    return frozen_graph, output_names


def convert_tf_rt(frozen_graph, output_names, saved_model_name, trt_dir):

    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )

    graph_io.write_graph(trt_graph, trt_dir, saved_model_name, as_text=False)


if __name__ == '__main__':

    # Clear any previous session.
    tf.keras.backend.clear_session()

    save_tf_dir = os.path.join(CUR_DIR, 'utils')
    model_path = os.path.join(CUR_DIR, 'utils', 'gender_model.h5')
    save_tf_model_name = 'gender_frozen_model.pb'
    save_trt_model_name = 'gender_trt_graph.pb'

    fr_graph, output_name = convert_h5_to_pb(model_p=model_path, save_pb_dir=save_tf_dir, pb_name=save_tf_model_name)
    convert_tf_rt(frozen_graph=fr_graph, output_names=output_name, saved_model_name=save_trt_model_name,
                  trt_dir=save_tf_dir)
