import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# data load
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# check nan data
print(np.isnan(x_train).any())
print(np.isnan(y_train).any())
print(np.isnan(x_test).any())
print(np.isnan(y_test).any())

# data preprocessing
input_shape = (28, 28, 1)
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train=x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test=x_test/255.0
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

# change data dimension method 1
# single_input = np.expand_dims(x_test[100], axis=0)
# single_input = np.expand_dims(y_test[100], axis=0)

# change data dimension method 2
single_input = tf.expand_dims(x_test[100], axis=0)
single_answer = tf.expand_dims(y_test[100], axis=0)

# load model
model = tf.keras.models.load_model('mnist_model.h5')

# prediction
Y_pred = model.predict(single_input)

# print result of prediction
Y_pred_classes = np.argmax(Y_pred)
Y_true = np.argmax(single_answer)
print("Predicted value", Y_pred_classes,", Ground Truth is", Y_true)

#check Y_pred data
print("Y_pred check")
print(Y_pred)

#draw result
x_value = [0,1,2,3,4,5,6,7,8,9]
plt.bar(x_value, Y_pred[0])
plt.xticks(x_value)
plt.savefig("result.png")