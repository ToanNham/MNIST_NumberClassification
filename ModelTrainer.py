# Followed instruction on this guide: https://tinyurl.com/2hbx5a24
 
import tensorflow as tf
import numpy as np

# Load dataset
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)= mnist.load_data()

# View an image from dataset
#import matplotlib.pyplot as plt
#plt.imshow(x_test[0], cmap=plt.get_cmap('gray'))
#plt.show()

# Import all required packages
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Dropout,Flatten,MaxPooling2D

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value
x_train /= 255


# Setting up the model
model = Sequential() 
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(200,activation = tf.nn.relu))
model.add(Dropout(0.3))
model.add(Dense(10,activation=tf.nn.softmax))

# Training the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x= x_train , y = y_train , epochs= 10)


# Saving to file
import pickle
f_out = open("model.bin", 'wb')
pickle.dump(model, f_out)
f_out.close()