from keras.models import load_model
import sys
import tensorflow as tf
import numpy as np
import glob
import os


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')
from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('C:/AirsimProject/AirSim/AutonomousDrivingCookbook/AirSimE2EDeepLearning/model/models/*.h5')
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model

print('Using model {0} for testing.'.format(MODEL_PATH))

model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')

car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))


def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)

    return image_rgba[76:135, 0:255, 0:3].astype(float)




count = 0
successCount = 0
successRate = 0


while (True):
    client.enableApiControl(True)
    car_state = client.getCarState()
    car_controls.brake = 0.0
    count += 1

    if (car_state.speed > 0.3):
        successCount += 1
    if (car_state.speed > 3.0):
        car_controls.brake = 1.0
    if (car_state.speed < 5):
        car_controls.throttle = 0.5
    else:
        car_controls.throttle = 0.0

    successRate = round((successCount / count) * 100, 2)

    image_buf[0] = get_image()
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)

    print('Sending steering = {0}, throttle = {1}, rate = {2}%'.format(car_controls.steering, car_controls.throttle, successRate))

    if (count > 100 and successRate < 89.8):
        client.reset()
        successRate = 0
        successCount = 0
        count = 0

    client.setCarControls(car_controls)


