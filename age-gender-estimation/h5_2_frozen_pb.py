# ref:
#   https://medium.com/@sebastingarcaacosta/
#   how-to-export-a-tensorflow-2-x-keras-model-to-a-frozen-and-optimized-graph-39740846d9eb

import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np

#tf.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()

#path of the directory where you want to save your model
frozen_out_path = './tensorflow_2_dlc'

model = tf.keras.models.load_model(sys.argv[1])

frozen_graph_filename = os.path.basename(sys.argv[1])

# Convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name="{}.pb".format(frozen_graph_filename),
                  as_text=False)

# Save its text representation
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name="{}.pbtxt".format(frozen_graph_filename),
                  as_text=True)

