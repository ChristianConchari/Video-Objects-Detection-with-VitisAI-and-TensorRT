import tensorflow as tf
#tf.compat.v1.disable_v2_behavior()
import numpy as np
from tensorflow import keras

def process_dataset():
    # Import the data
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    x_train = np.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = np.reshape(x_test, (NUM_TEST, 28, 28, 1))
    return x_train, y_train, x_test, y_test

def retrieve_model():
    loaded_model = tf.keras.models.load_model("models/fruits_model.h5")
    # loaded_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001),metrics=['accuracy'])
    return loaded_model

def save(model, filename):
    # First freeze the graph and remove training nodes.
    output_names = model.output.op.name
    sess = tf.compat.v1.keras.backend.get_session()
    frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_names])
    frozen_graph = tf.compat.v1.graph_util.remove_training_nodes(frozen_graph)
    # Save the model
    with open(filename, "wb") as ofile:
        ofile.write(frozen_graph.SerializeToString())

def main():

    model = retrieve_model()
    
    model.summary()
    model.save('own_model')

    # Evaluate the model on test data
    #save(model, filename="models/own_opt_model.pb")

if __name__ == '__main__':
    main()

