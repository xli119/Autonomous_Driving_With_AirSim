from keras.models import load_model
import sys
import numpy as np
import glob
import os
import cv2
import tensorflow as tf
if (r'E:\endtoend\PythonClient' not in sys.path):
    sys.path.insert(0, r'E:\endtoend\PythonClient')
from AirSimClient import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

count = 0
successCount = 0
successRate = 0

if (MODEL_PATH == None):
    models = glob.glob('model/models/*.h5')
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

def deshadow(ori):
    y_cb_cr_img = cv2.cvtColor(ori, cv2.COLOR_BGR2YCrCb)
    binary_mask = np.copy(y_cb_cr_img)
    y_mean = np.mean(cv2.split(y_cb_cr_img)[0])
    y_std = np.std(cv2.split(y_cb_cr_img)[0])
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):
            if y_cb_cr_img[i, j, 0] < y_mean - (y_std / 3):
                binary_mask[i, j] = [255, 255, 255]
            else:
                binary_mask[i, j] = [0, 0, 0]
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(binary_mask, kernel, iterations=1)
    spi_la = 0
    spi_s = 0
    n_la = 0
    n_s = 0
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):
            if erosion[i, j, 0] == 0 and erosion[i, j, 1] == 0 and erosion[i, j, 2] == 0:
                spi_la = spi_la + y_cb_cr_img[i, j, 0]
                n_la += 1
            else:
                spi_s = spi_s + y_cb_cr_img[i, j, 0]
                n_s += 1
    average_ld = spi_la / n_la
    average_le = spi_s / n_s
    i_diff = average_ld - average_le
    ratio_as_al = average_ld / average_le
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):
            if erosion[i, j, 0] == 255 and erosion[i, j, 1] == 255 and erosion[i, j, 2] == 255:
                y_cb_cr_img[i, j] = [y_cb_cr_img[i, j, 0] + i_diff, y_cb_cr_img[i, j, 1] + ratio_as_al,
                                     y_cb_cr_img[i, j, 2] + ratio_as_al]
    image = cv2.cvtColor(y_cb_cr_img, cv2.COLOR_YCR_CB2BGR)
    return image


while (True):
    client.enableApiControl(True)
    car_state = client.getCarState()
    car_controls.brake = 0.0
    count += 1
    if (car_state.speed >0.3):
        successCount += 1
    if (car_state.speed > 3.0):
        car_controls.brake = 1.0
    if (car_state.speed < 5):
        car_controls.throttle = 0.5
    else:
        car_controls.throttle = 0.0
    successRate = round((successCount/count)*100,2)
    imagetemp = get_image()

    image_buf[0] = imagetemp
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)

    print('Sending steering = {0}, throttle = {1}, rate = {2} %'.format(car_controls.steering, car_controls.throttle,successRate))

    if(count >100 and successRate < 89.8) :
        client.reset()
        successRate = 0
        successCount = 0
        count = 0

    client.setCarControls(car_controls)
