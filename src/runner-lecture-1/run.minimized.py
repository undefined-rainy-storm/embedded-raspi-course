import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import config as c

test_data = pd.read_csv(c.PATH_DATAFRAME_TEST_CSV)
answer_data = pd.read_csv(c.PATH_DATAFRAME_GROUNDTRUTH_CSV)
test_data = test_data.to_numpy()
answer_data = answer_data.to_numpy()

test_data = test_data.reshape(100,)
answer_data = answer_data.reshape(100,)

model = tf.lite.Interpreter(c.PATH_MODEL_BINARY_QUANTIZED)
model.allocate_tensors()

input_index = model.get_input_details()[0]['index']
output_index = model.get_output_details()[0]['index']

predict = []

for x_value in test_data:
    x_value_tensor = tf.convert_to_tensor([[x_value]], dtype = np.float32)
    model.set_tensor(input_index, x_value_tensor)
    model.invoke()
    predict.append(model.get_tensor(output_index)[0])
    plt.clf()

plt.title('Comparison of various models against actual values')
plt.plot(test_data, answer_data, 'bo', label = 'Actual')
plt.plot(test_data, predict, 'g^', label = 'Quant predictions')
plt.legend()
plt.show()
