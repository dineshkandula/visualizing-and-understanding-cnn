import sys
import numpy as np

import dlogging

from dmodel import *

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.applications import vgg16

from PIL import Image

LOGGING_ENABLED = False
dlog = dlogging.dlogger("VISUALIZE_CNN_TC", dlogging.DEBUG, LOGGING_ENABLED)

def un_activation(layer, data):
    '''
    un_activation(layer, data)

    Data is expected to have the sample co-ordinate.
    '''
    activation = layer.activation
    return activation(data)

def un_max_pooling(model, layer_name, data, switch):
    '''
    un_max_pooling(model, layer_name, data, switch)

    Data is expected to have the sample co-ordinate.
    switch is of same dimensions as data. Each entry in switch
    has the coordinates of the point where max came from.
    '''
    layer = dget_layer(model, layer_name)
    pool_size = layer.pool_size
    strides = layer.strides

    if pool_size != strides:
        sys.exit("pool size is not equal to strides")

    size = (data.shape[0], data.shape[1] * pool_size[0], data.shape[2] * pool_size[1], data.shape[3])
    output = np.zeros(size, dtype=float)

    for s in range (switch.shape[0]):
        for i in range (switch.shape[3]):
            for j in range (switch.shape[1]):
                for k in range (switch.shape[2]):
                    r=switch[s, j, k, i, 0]
                    c=switch[s, j, k, i, 1]
                    output[s, r, c, i] = data[s, j, k, i]

    return output

def get_switch(data, pool_size):
    '''
    get_switch(data, pool_size)

    Data is expected to have sample coordinate.

    Returns switch array which stores the location of the max of all the windows.
    This function assumes stride size is same as pool size in both the directions.
    '''

    data_shape = data.shape
    output_size = (data_shape[0], data_shape[1]//pool_size[0], data_shape[2]//pool_size[1], data_shape[3], 2)
    output = np.zeros(output_size, dtype=int)

    for i in range(output_size[0]): #traverse through samples
        for j in range(output_size[3]): #traverse through channels
            for r in range(output_size[1]):
                for c in range(output_size[2]):
                    window = data[i, (r*pool_size[0]):((r+1)*pool_size[0]),
                                                (c*pool_size[1]):((c+1)*pool_size[1]), j]
                    mx = np.argmax(window)
                    dlog.debug(window, channels_first=True)
                    dlog.debug(mx)
                    output[i][r][c][j][0] = r*pool_size[0] + (mx//window.shape[0])
                    output[i][r][c][j][1] = c*pool_size[1] + (mx % window.shape[0])

    return output

def unFilter(data, model, layer_name):
    '''
    unFilter(data, model, layer_name)

    Returns the output obtained by applying inverted filtering operation on
    the feature map.

    Data is single sample. It shouldn't have sample coordinate.

    and the layer should be conv2D layer.
    '''

    #### get weights ####
    #0 bias will be used during invert opeartion
    W = dget_weights(model, layer_name)[0]
    # Interchange channels and filters dimensions.
    W = W.transpose(0, 1, 3, 2)

    #flip rows and columns vertically and horizontally
    W = W[::-1, ::-1, :, :]    
    B = np.zeros(W.shape[3])

    #create a model with W, B as weights and data as input to calculate the output
    dlog.debug(W.shape)
    dlog.debug(data.shape)

    inv_model = keras.Sequential(
        [
            keras.Input(shape = data.shape),
            layers.Conv2D(W.shape[3], kernel_size = (W.shape[0], W.shape[1]), kernel_initializer=tf.constant_initializer(W),
                    bias_initializer=tf.constant_initializer(B), padding='same')
        ]
    )

    # dset_weights(inv_model, 0, W, B)
    inp = np.expand_dims(data, axis=0)
    output = inv_model.predict(inp)
    return output

def get_feature_for_visualization(output, feature, mode='all'):
    feature_map = None
    if mode == 'all':
        feature_map = output[:,:, :, feature]
    elif mode == 'max':
        feature_map = output[:, :, :, feature]
        max_pos = dget_max_location_2d(feature_map[0, :, :])
        temp = np.zeros_like(feature_map)
        temp[0, max_pos[0], max_pos[1]] = feature_map[0, max_pos[0], max_pos[1]]
        feature_map = temp

    ret = np.zeros_like(output)
    ret[:, :, :, feature] = feature_map
    return ret

def deconv_feature(img, model, layer_name, feature, mode='all'):
    input = model.input
    outputs = [layer.output for layer in model.layers]
    func = dget_function([input], outputs)
    output_list = func(img)
    dlog.debug(len(output_list))

    layer_idx = dget_layer_index_from_name(model, layer_name)
    dlog.debug(layer_idx, flag=True)

    curr_output = get_feature_for_visualization(output_list[layer_idx], feature, mode)

    while layer_idx>0:
        dlog.debug("layer idx: {}", layer_idx, flag=True)
        dlog.debug("curr shape: {}", curr_output.shape, flag=True)

        curr_layer = model.layers[layer_idx]
        if dis_layer_conv2d(curr_layer):
            dlog.debug("conv2d layer", flag=True)
            if curr_layer.get_config().get('activation') == 'relu':
                curr_output = un_activation(curr_layer, curr_output)

            curr_output = unFilter(curr_output[0, :, :, :], model, curr_layer.name)
        elif dis_layer_maxpooling2d(curr_layer):
            data = output_list[layer_idx-1]
            switch = get_switch(data, curr_layer.pool_size)    
            curr_output = un_max_pooling(model, curr_layer.name, curr_output, switch)
        elif dis_layer_flatten(curr_layer):
            pass
        elif  dis_layer_dense(curr_layer):
            pass
        elif dis_layer_dropout(curr_layer):
            pass

        layer_idx = layer_idx - 1

    dlog.debug(curr_output.shape, flag=True)
    return curr_output[0, :, :, :]

def post_process_visualization(vis):
    dlog.debug(vis.shape, flag=True)
    vis = vis - vis.min()
    vis *= 1.0 / (vis.max() + 1e-8)

    uint8_deconv = (vis * 255).astype(np.uint8)
    img = Image.fromarray(uint8_deconv, 'RGB')
    dlog.imshow(img, flag=True)

def visualize(input, model, layer_name, feature, mode='all'):
    '''
    visualize(img, model, layer_name, feature, mode='all')

    Implements deconv net and displays the feature.
    '''
    img = deconv_feature(input, model, layer_name, feature, mode)
    post_process_visualization(img)
