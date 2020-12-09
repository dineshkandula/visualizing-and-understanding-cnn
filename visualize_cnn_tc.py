import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.imagenet_utils import decode_predictions

import dlogging
from dlogging import dlogger

from dmodel import *

from visualize_cnn import un_activation, un_max_pooling, un_filter, get_switch

LOGGING_ENABLED = True
dlog = dlogger("VISUALIZE_CNN_TC", dlogging.DEBUG, LOGGING_ENABLED)

TEST_IMAGE_PATH = '../../../datasets/test/cat.png'

def un_activation_test():
    vgg_model = vgg16.VGG16(weights='imagenet')
    vgg_model.summary()
    rand_data = np.random.randint(low=-10, high=10, size=(1, 112, 112, 128))
    dlog.debug(rand_data)
    rand_data = un_activation(dget_layer(vgg_model, "block2_conv2"), rand_data)
    dlog.debug(rand_data)

def get_switch_test():
    A = np.random.randint(low=-10, high=10, size=(1, 4, 4, 4))
    dlog.debug(A[:, :, :, 1])
    R = get_switch(A, (2, 2))
    dlog.debug(R.shape)
    dlog.debug(R[:, :, :, 1, 0])
    dlog.debug(R[:, :, :, 1, 1])


def max_pool(data, pool_size):
    output_size = (data.shape[0], data.shape[1]//pool_size[0], data.shape[2]//pool_size[1], data.shape[3])
    output = np.zeros(output_size, dtype=int)

    for i in range(output.shape[0]):
        for j in range(output.shape[3]):
            for r in range(output.shape[1]):
                for c in range(output.shape[2]):
                    window = data[i, (r*pool_size[0]):((r+1)*pool_size[0]),
                                                (c*pool_size[1]):((c+1)*pool_size[1]), j]
                    mx = np.amax(window)
                    output[i, r, c, j] = mx
    
    return output


def un_max_pooling_test():
    A = np.random.randint(low=-10, high=10, size=(1, 4, 4, 4))
    dlog.debug(A, flag=True, channels_first=True)
    R = get_switch(A, (2, 2))
    vgg_model = vgg16.VGG16(weights='imagenet')
    data = max_pool(A, (2, 2))
    output = un_max_pooling(vgg_model, 'block2_pool', data, R)
    dlog.debug(output, flag=True, channels_first=True)

def un_filter_test():
    vgg_model = vgg16.VGG16(weights='imagenet')
    inter_model = dget_intermediate_model(vgg_model, 'block3_conv3')

    # load test image
    img = load_img(TEST_IMAGE_PATH, target_size=(224, 224))
    numpy_image = img_to_array(img)
    image_batch = np.expand_dims(numpy_image, axis=0)
    # prepare the image for the VGG model
    processed_image = vgg16.preprocess_input(image_batch)
    pred = inter_model.predict(processed_image)
    pred = np.squeeze(pred)
    output = un_filter(pred, inter_model, 'block3_conv3')
    dlog.debug(output)

un_activation_test()
get_switch_test()
un_max_pooling_test()
un_filter_test()
