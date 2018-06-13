"""author: Xiaoyu Li
Created on 6/12/2018
To calculate average running time of each test in landscape mountain"""

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

if ("\PythonClient" not in sys.path):
    sys.path.insert(0, "\PythonClient")
from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('D:\EndToEndLearningRawData\model_output\models\*.h5')
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model

print('Using model {0} for testing.'.format(MODEL_PATH))


model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
time0 = time.time()
start_time = time.time()
time_list = list()
print('Connection established!')


car_controls.steering = 0
car_controls.throttle = 0.6
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))


def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)

    return image_rgba[76:135, 0:255, 0:3].astype(float)


while (True):
    car_state = client.getCarState()
    car_collision = client.getCollisionInfo()

    car_controls.brake = 0
    if (car_state.speed < 3.0):
        car_controls.throttle = 0.6
    elif (car_state.speed > 4.0):
        car_controls.brake = 1.0


    else:
        car_controls.throttle = 0.6


    if car_collision.has_collided or car_state.speed == 0:
        end_time = time.time()
        time1 = end_time - start_time
        start_time = time.time()
        if time1 > 5:
            time_list.append(time1)
        time.sleep(1.5)
        client.reset()
        time.sleep(1)

        if end_time - time0 > 60:
            sum = 0
            for t in time_list:
                sum += t
            print("Average Running Time:", sum / len(time_list))
            print(time_list)
            break
            sys.stdout.flush()




    image_buf[0] = get_image()

    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])

    model_output = model.predict([image_buf, state_buf])

    car_controls.steering = round(0.4 * float(model_output[0][0]), 2)

    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))


    client.setCarControls(car_controls)