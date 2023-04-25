'''
Define 1D ResNet objects.
'''

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, MaxPool1D, Dense

from tensorflow_addons.layers import AdaptiveAveragePooling1D

def ResBlock1D(in_c, out_c, kernel_size, stride1=1, stride2=1, padding='same', downsample=None) :
    '''
    Defines a residual block for a 1D CNN.
    '''
    
    # initialize input
    X = Input(shape=(None, in_c))
    inp = X
    
    # conv1
    out = Conv1D(out_c, kernel_size, strides=stride1, padding=padding)(X)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    
    # conv2
    out = Conv1D(out_c, kernel_size, strides=stride2, padding=padding)(out)
    out = BatchNormalization()(out)
    
    # downsample if specified
    if downsample is not None :
        X = downsample(X)
        
    # residual connection
    out += X
    out = ReLU()(out)
    
    return Model(inp, out)

def ResNet18_1D(lat_dim=32) :
    '''
    Defines a 1D ResNet architecture based on ResNet18.
    '''
    
    # add a channel dimension to input
    inp = Input(shape=(1000))
    X = inp[..., tf.newaxis]
    
    # first conv maps to latent space (downsample 2x)
    out = Conv1D(lat_dim, 7, strides=2, padding='same')(X)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    out = MaxPool1D()(out)
    
    # two residual blocks which maintain shape
    out = ResBlock1D(lat_dim, lat_dim, 3)(out)
    out = ResBlock1D(lat_dim, lat_dim, 3)(out)
    
    # two residual blocks with downsampling
    downsample1 = Sequential([
        Conv1D(lat_dim*2, 1, strides=2, padding='same', use_bias=False),
        BatchNormalization()
    ])
    out = ResBlock1D(lat_dim, lat_dim*2, 3, stride1=2, downsample=downsample1)(out)
    out = ResBlock1D(lat_dim*2, lat_dim*2, 3)(out)
    
    # two residual blocks with downsampling
    downsample2 = Sequential([
        Conv1D(lat_dim*4, 1, strides=2, padding='same', use_bias=False),
        BatchNormalization()
    ])
    out = ResBlock1D(lat_dim*2, lat_dim*4, 3, stride1=2, downsample=downsample2)(out)
    out = ResBlock1D(lat_dim*4, lat_dim*4, 3)(out)
    
    # two residual blocks with downsampling
    downsample3 = Sequential([
        Conv1D(lat_dim*8, 1, strides=2, padding='same', use_bias=False),
        BatchNormalization()
    ])
    out = ResBlock1D(lat_dim*4, lat_dim*8, 3, stride1=2, downsample=downsample3)(out)
    out = ResBlock1D(lat_dim*8, lat_dim*8, 3)(out)
    
    # average pooling and linear
    out = AdaptiveAveragePooling1D(1)(out)
    out = Dense(1)(out)
    
    return Model(inp, out)