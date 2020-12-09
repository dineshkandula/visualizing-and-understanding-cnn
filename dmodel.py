import numpy as np

from tensorflow import keras
import keras.backend as K


def dget_layer(model, layer_name):
    '''
    get_layer(model, layer_name)

    Returns layer in the @p model with layer name @p layer_name.
    '''

    return model.get_layer(layer_name)

def dget_intermediate_model(model, layer_name):
    '''
    get_intermediate_model(model, layer_name)

    Returns a model with input of @p model as input and output as 
    a layer with name @p layer_name.
    '''
    return keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

def dget_function(input, output):
    '''
    get_function(input, output)

    Returns function with given input list and output list as input and output respectively.
    '''
    return K.function(input, output)

def dget_weights(model, layer_name):
    '''
    dget_weights(model, layer_name)

    Returns tuple of weights and biases.
    '''    

    return dget_layer(model, layer_name).get_weights()

def dset_weights(model, layer, weights, biases):
    if isinstance(layer, str):
        model.get_layer(layer).set_weights([weights, biases])
    elif isinstance(layer, int):
        model.layers[layer].set_weights([weights, biases])

def dget_layer_index_from_name(model, layer_name):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layer_name:
            index = idx
            break

    return index

def dget_max_location_2d(arr):
    assert arr.ndim == 2
    max=-np.inf
    pos=None
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if max < arr[i,j]:
                max = arr[i, j]
                pos=(i, j)

    return pos

def dis_layer_conv2d(layer):
    return isinstance(layer, keras.layers.Conv2D)

def dis_layer_maxpooling2d(layer):
    return isinstance(layer, keras.layers.MaxPooling2D)

def dis_layer_flatten(layer):
    return isinstance(layer, keras.layers.Flatten)

def dis_layer_dense(layer):
    return isinstance(layer, keras.layers.Dense)

def dis_layer_dropout(layer):
    return isinstance(layer, keras.layers.Dropout)

