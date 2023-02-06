import pickle
import numpy as np
import tensorflow as tf

# Loading the data
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()

# Pre-processing the images
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255

# Loading the model
model = pickle.load(open('model.bin','rb'))

# Testing the model
model.evaluate(x_test, y_test)