import keras
import tensorflow as tf 
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras.models import load_model, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D, Conv2DTranspose, Reshape, concatenate, BatchNormalization, Activation
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.models import Sequential


def rmse_k(y_true, y_pred):
    return keras.backend.sqrt(keras.backend.mean(keras.backend.square(y_pred - y_true), axis=-1))

#Let's define a basic CNN with few convolutions and MaxPooling
def block_conv(conv, filters):
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(filters, 3, padding='same')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    return conv

def block_up_conc(conv, filters,conv_conc):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = concatenate([conv,conv_conc])
    conv = block_conv(conv, filters)
    return conv

def block_up(conv, filters):
    conv = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
    conv = block_conv(conv, filters)
    return conv


def highestPowerof2(n):
    res = 0;
    for i in range(n, 0, -1):
         
        # If i is a power of 2
        if ((i & (i - 1)) == 0):
         
            res = i;
            break;
         
    return res;


def unet_maker( nb_inputs,size_target_domain,shape_inputs, filters = 64, OROG=False,seed=123):
    from math import log2,pow
    import os
    import numpy as np
    import random as rn
    inputs_list=[]
    size=np.min([highestPowerof2(shape_inputs[0][0]),highestPowerof2(shape_inputs[0][1])])
   
    
    if nb_inputs==1:
        inputs = keras.Input(shape = shape_inputs[0])
        conv_down=[]
        diff_lat=inputs.shape[1]-size+1
        diff_lon=inputs.shape[2]-size+1
        conv0=Conv2D(32, (diff_lat,diff_lon))(inputs)
        conv0=BatchNormalization()(conv0)
        conv0=Activation('relu')(conv0)
        prev=conv0
        for i in range(int(log2(size))):
            conv=block_conv(prev, filters*int(pow(2,i)))
            pool=MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv)
            conv_down.append(conv)
            prev=pool
        up=block_conv(prev, filters*int(pow(2,i)))
        k=log2(size)
        for i in range(1,int(log2(size_target_domain)+1)):
            if i<=k:
                up=block_up_conc(up,filters*int(pow(2,k-i)),conv_down[int(k-i)])
            else :
                up=block_up(up,filters)
        inputs_list.append(inputs)     
                
    if nb_inputs==2:
        inputs = keras.Input(shape = shape_inputs[0])
        conv_down=[]
        diff_lat=inputs.shape[1]-size+1
        diff_lon=inputs.shape[2]-size+1
        conv0=Conv2D(32, (diff_lat,diff_lon))(inputs)
        conv0=BatchNormalization()(conv0)
        conv0=Activation('relu')(conv0)
        prev=conv0
        for i in range(int(log2(size))):
            conv=block_conv(prev, filters*int(pow(2,i)))
            pool=MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(conv)
            conv_down.append(conv)
            prev=pool
        
        last_conv=block_conv(prev, filters*int(pow(2,i)))
        inputs2 = keras.Input(shape=shape_inputs[1])
        model2 = Dense(filters)(inputs2)
        for i in range(1,int(log2(size))):
            model2 = Dense(filters*int(pow(2,i)))(model2)
    
        merged = concatenate([last_conv,model2])
        #up=concatenate([pool_last,conv_down[int(log2(size_input_domain)-1)]])
        up=merged
        k=log2(size)
        for i in range(1,int(log2(size_target_domain)+1)):
            if i<=k:
                up=block_up_conc(up,filters*int(pow(2,k-i)),conv_down[int(k-i)])
            else :
                conv=block_up(up,filters)
                up=conv
        inputs_list.append(inputs)
        inputs_list.append(inputs2)
    if OROG:
        inputs3 = keras.Input(shape= [64,64,1])
        conv_orog = block_conv(inputs3, filters)
        concate=concatenate([up,conv_orog])
        last=block_conv(concate, filters)
        inputs_list.append(inputs3)
    else:
        last=up
        
    lastconv=Conv2D(1, 1, padding='same')(last)
    return (keras.models.Model(inputs=inputs_list, outputs=lastconv))
