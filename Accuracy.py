import h5py
from keras.models import load_model
import sys
import numpy as np
import glob
import os
import tensorflow as tf
import time
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


test_dataset = h5py.File('D:\EndToEndLearningRawData\data_cooked_nh/test.h5', 'r')

num_test_example = test_dataset['image'].shape[0]
print(test_dataset['previous_state'].shape)

if ("\PythonClient" not in sys.path):
    sys.path.insert(0, "\PythonClient")
from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('D:/EndToEndLearningRawData/model_final_nh/nROI60140Drop=0.0_bright=0.9/models/*.h5')
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model

print('Using model {0} for testing.'.format(MODEL_PATH))

model = load_model(MODEL_PATH)

image_buf = np.zeros((1, 80, 255, 3))
state_buf = np.zeros((1,4))

for i in range(num_test_example):
    current_image = test_dataset['image'][i]
    # image1d = np.fromstring(current_image.image_data_uint8, dtype=np.uint8)
    # image_rgba = current_image.reshape(current_image.height, current_image.width, 4)
    image_buf[0] = current_image[60:140, 0:255, 0:3].astype(float)
    state_buf[0] = np.array([test_dataset['previous_state'][i][0], test_dataset['previous_state'][i][1],
                             test_dataset['previous_state'][i][2], test_dataset['previous_state'][i][3]])

    model_output = model.predict([image_buf, state_buf])
    #print(test_dataset['label'][i], test_dataset['previous_state'][i][0])
    #print("steering_true:", test_dataset['label'][i], 'steering_pred:', model_output[0][0])
    print("steering_true:", test_dataset['previous_state'][i+1][0], 'steering_pred:', model_output[0][0])


