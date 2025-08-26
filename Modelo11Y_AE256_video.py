#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 21:22:43 2021

@author: nautec
"""


"""
Created on Thu Apr  2 18:13:40 2020
@author: Claudio Mello

Implementacao Autoencoder em keras. 
Inicializacao da rede usando "GlorotNormal"
Otimizador Adam
Dataset  UWData_Paired_Sat080 (2100 imagens .png underwater de 128x128, formato "channels last")


"""
# TF modules
import tensorflow as tf
import tensorflow_probability as tfp
#
# Keras modules
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import LeakyReLU, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
#
# Other modules
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import numpy as np
import os
#
# Model utils
from fgaussian import gaussian
from funcs_model_test_video2 import get_median
from funcs_model_test_video2 import vals
from funcs_model_test_video2 import valsr
from funcs_model_test_video2 import cena
from funcs_model_test_video2 import haze
from funcs_model_test_video2 import param2
from funcs_model_test_video2 import param_comp
from funcs_model_test_video2 import percentil


time_format = "%H:%M:%S"
walltime = datetime.now()
begin_time = walltime.strftime(time_format)

# Directorys
wdir = os.getcwd()
work_dir = wdir
output_dir = os.path.join(work_dir, "outputs") # output directory
#
img_dataset_dir = os.path.join(work_dir,"datasets/images")
validation_image_dataset_dir = os.path.join(work_dir,"datasets/images")
#
video_dataset_dir = os.path.join(work_dir,"datasets/videos")
raw_video_dir = ""
#
model_dir = os.path.join(work_dir,"models/checkpoints")
model_name = "DENSERcDB_Full.json"
loss_curve = "DENSERcDB_Full_Loss"

# ============================================================================
train_dataset =  os.path.join(img_dataset_dir,"UWData256_2K2")

with open(train_dataset, 'rb') as f:        #  <= 2200 imagens
    data = pickle.load(f)


test_dataset = os.path.join(validation_image_dataset_dir,"UWTest256X")
with open(test_dataset, 'rb') as f:
    seleta = pickle.load(f)

seleta = np.clip(seleta, 0.001, 1.0)


# dimensions of our images.
ni, ht, wd, ch = np.shape(data)

input_shape = (ht, wd, ch)


#=============================================================================
#latent_dim = 32
fe1 = [48, 48, 36, 36]   #48, 48, 36, 32
fe2 = [48, 48, 36, 32]   #48, 48, 36, 32
fd = [36, 36, 48, 48]    #64, 48, 36, 32
alpha = 0.19
alpha0 = 0.025
alpha1 = 0.09
eps = 1.0e-7
seedy1 = 19    ##17
seedy2 = 35    ##31
seedy3 = 43    ##43
uu = 0.02e-6
kkr = 15e-6  
bbr = 1.5e-6
regk = regularizers.l1(kkr)
regb = regularizers.l1(bbr)

epochs = 200
batch_size = 6


def escala(img):
    """Escale function"""

    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    a = tf.constant([-1.0], dtype=float)
    vm = K.clip(tf.math.multiply(a, vmin), 0., 1.0)
    #vm = K.clip(vm, 0., 0.3)
    imgx = tf.math.add(vm, img)
    nmax = tf.reduce_max(imgx, axis=(1,2), keepdims=True)
    nmax = K.clip(nmax, 1.0, vmax)
    imgx = tf.math.divide_no_nan(imgx, nmax)
    
    return imgx

def write_file(file, name, path):
    """Persists output file"""

    name = path + '/' + name
    with open(name, "wb") as f:
        pickle.dump(file, f)

 
def sig_in(img):
    """sig"""

    um = tf.constant([1.0], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    m5 = tf.constant([-5.0], dtype=float)
    z5 = tf.constant([0.5], dtype=float)    
    fsig = tf.math.divide(um, tf.math.add(um, tf.math.exp(tf.math.multiply(m5, tf.math.subtract(img, z5)))))
    img = tf.math.divide(tf.math.add(img, fsig), dois)
    
    return img
       

# ==============================  AUTOENCODER ================================

#Encoder ----------------------------------------------------------
#Bloco 256 --------------------------------------------------------

input_img = Input(shape=input_shape, dtype=tf.float32, name='Entrada')
#x1 = Lambda(sig_in, name = 'AjusteSig')(input_img)

x1 = Conv2D(fe1[0],
    (3,3),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(input_img) 

# xp1 = Conv2D(fe1[0],
#     (4,4),
#     strides=(1, 1),
#     padding="same",
#     data_format=None,
#     dilation_rate=(1, 1),
#     activation='relu',
#     use_bias=True,
#     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
#     bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
#     kernel_regularizer=regk,
#     bias_regularizer=regb,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
# )(input_img) 
# x1 = Concatenate()([x1, xp1])

x1 = Conv2D(fe1[0],
    (3,3),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1) 

x1 = Conv2D(fe1[0],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1) 
x256e1 = x1


#Fim do Bloco 256 --------------------------------------------------------

#Bloco 128 --------------------------------------------------------
x1 = Conv2D(fe1[0],
    (2,2),
    strides=(2, 2),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)    #128


x1 = Conv2D(fe1[0],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)    #128

x1 = Conv2D(fe1[0],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)           #128


#x1 = BatchNormalization(axis=-1, epsilon = eps)(x1)

x1 = Conv2D(fe1[0],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)           #128
x128e1 = x1

#FIM bloco 128-e1 ---------------------------------------------

# xc2a = x1    #   <--------------Para a concatenacao 

# Bloco 64-e1 -------------------------------------------------
x1 = Conv2D(fe1[1],
    (2,2),
    strides=(2, 2),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

x1 = Conv2D(fe1[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+5),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+5),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

x1 = Conv2D(fe1[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+6),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+6),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

#x1 = BatchNormalization(axis=-1, epsilon = eps)(x1)


x1 = Conv2D(fe1[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)      #64x64
x64e1 = x1

#FIM bloco 64-e1 ----------------------------------------------------


#Bloco 32-e1 --------------------------------------------------------
x1 = Conv2D(fe1[2],
    (2,2),
    strides=(2, 2),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)       #32

x1 = Conv2D(fe1[2],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

x1 = Conv2D(fe1[2],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)

x1 = Conv2D(fe1[2],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
)(x1)
x32e1 = x1

#FIM bloco 32-e1 --------------------------------------------------

#Bloco 16-e1 ------------------------------------------------------
x1 = Conv2D(fe1[3],
    (2,2),
    strides=(2, 2),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+4),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(x1)       #16
x16e1 = x1

x1 = Conv2D(fe1[3],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+5),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+5),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(x1)       #16

x16e2 = x1

# x1 = Conv2D(fe1[3],
#     (1,1),
#     strides=(1, 1),
#     padding="same",
#     data_format=None,
#     dilation_rate=(1, 1),
#     activation='relu',
#     use_bias=True,
#     kernel_initializer="glorot_normal",
#     bias_initializer="glorot_normal",
#     kernel_regularizer=regularizers.l1(uu),
#     bias_regularizer=regularizers.l1(uu),
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None
# )(x1)       #16

x1 = Conv2D(fe1[3],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+6),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy1+6),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(x1)       #16

x1 = Concatenate()([x1, x16e1, x16e2])           #<<<<<<<Concatenacao

xd = x1

##############################################################################
#Decoder  -----------------------------------------------------
#Bloco 16-d ---------------------------------------------------
xd = Conv2D(fd[0],
    (2,2),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[0],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Concatenate()([xd, x16e1])
xd = Conv2D(fd[0],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

#xd = BatchNormalization(axis=-1, epsilon = eps)(xd)
# Fim do bloco 16-d -----------------------------------------------------------

#Bloco 32-d ---------------------------------------------------

xd = UpSampling2D((2, 2))(xd)       #16 -> 32

#******* Switch conc 3 *******
# if random.randint(1,2) == 1:
#     xc32 = x32e1
# else: xc32 = x32e1
# xc32 = BatchNormalization(axis=-1, epsilon = eps)(xc32)
xd = Concatenate()([xd, x32e1])      #<<<<<<<<<<Concatenacao

xd = Conv2D(fd[1],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+6),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+6),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+4),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+4),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[1],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+5),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+5),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

# xd = BatchNormalization(axis=-1, epsilon = eps)(xd)

#FIM bloco 32-d -------------------------------------------------

xd = UpSampling2D((2, 2))(xd)       #32 -> 64

#Bloco 64-d ----------------------------------------------------

#******* Switch conc 2 *******
# if random.randint(1,2) == 1:
#     xc64 = x64e1
# else: xc64 = x64e1
xd = Concatenate()([xd, x64e1])

xd = Conv2D(fd[2],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+6),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+6),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[2],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[2],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+7),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+7),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[2],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+8),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+8),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

# xd = BatchNormalization(axis=-1, epsilon = eps)(xd)

#FIM bloco 64-d -------------------------------------------------

#Bloco 128-d ----------------------------------------------------
xd = UpSampling2D((2, 2))(xd)        #64 -> 128

#******* Switch conc 1 *******
# if random.randint(1,2) == 1:
#     xc128 = x128e1
# else: xc128 = x128e1
xd = Concatenate()([xd, x128e1])      #<<<<<<<<<<

xd = Conv2D(fd[3],
    (2,2),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)
#xd = LeakyReLU(alpha)(xd)

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+11),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+11),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)
#xd = LeakyReLU(alpha0)(xd)

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+1),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+1),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

#FIM bloco 128-d -------------------------------------------------------------

#Bloco 256 - d  --------------------------------------------------------------
xd = UpSampling2D((2, 2))(xd)        #64 -> 128

#xcc = xd
#skip 256
xd = Concatenate()([xd, x256e1])      #<<<<<<<<<<

xd = Conv2D(fd[3],
    (2,2),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+9),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)

#xd = Concatenate()([xd, xcc])

xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+11),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+11),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)


xd = Conv2D(fd[3],
    (1,1),
    strides=(1, 1),
    padding="valid",
    data_format=None,
    dilation_rate=(1, 1),
    activation='relu',
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+12),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+12),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)
#fim do bloco 256 - d --------------------------------------------------------

# xd = BatchNormalization(axis=-1, epsilon = eps)(xd)
xd = Conv2D(3,
    (2,2),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+2),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+2),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None
)(xd)
xd = LeakyReLU(alpha)(xd)

output_img0 = Conv2D(3,
    (1,1),
    strides=(1, 1),
    padding="same",
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+3),
    bias_initializer=tf.keras.initializers.GlorotNormal(seed=seedy3+3),
    kernel_regularizer=regk,
    bias_regularizer=regb,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    name="SD0"
)(xd)
output_img = LeakyReLU(alpha, name='SD1')(output_img0)

output_img = Lambda(escala, name = 'SD2')(output_img)

#================================= OTIMIZADOR ================================
opt = optimizers.Adam(lr = 0.00007, 
                      beta_1=0.9, 
                      beta_2=0.999, 
                      epsilon = 1.1e-7, 
                      amsgrad=False)
                      
# ============================================================================
# =========================== Calcula a mediana ==============================
def get_median_(ix):
    """Get median from ix"""

    median = tfp.stats.percentile(ix, 50.0, axis=(1,2),
                                  interpolation='midpoint',
                                  preserve_gradients=True)
    # median = K.reshape(median, (nbatch, 1, 1, 3))
    return median


def get_median2_(ix):
    """Get median from ix"""

    md = tfp.stats.percentile(ix, 50.0, axis=None,
                                  interpolation='midpoint',
                                  preserve_gradients=True)
    # median = K.reshape(median, (nbatch, 1, 1, 3))
    return md


def percentil_(img, per):
    """Percentil"""

    perc = tfp.stats.percentile(img, per, axis=(1,2),
                                  interpolation='midpoint',
                                  preserve_gradients=True)
    
    return perc


def vals_(img):
    """Image vals"""

    #nbatch = K.shape(img)[0]
    #vmdn = K.reshape(get_median(img), (nbatch, 1, 1, 3))
    dois = tf.constant([2.0], dtype=float)
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    vmed = tf.reduce_mean(img, axis=(1,2), keepdims=True)
    B = tf.math.divide(tf.math.add(vmax, vmin), dois)    
    bw = K.clip((vmax - vmin), 0.001, 1.0)
    
    return vmax, vmin, bw, vmed, B


def bw_adj(img):
    """Bw"""

    n15 = tf.constant([1.5], dtype=float)
    nbatch = K.shape(img)[0]
    p75 = K.reshape(percentil(img, 75.0), (nbatch, 1, 1, 3))
    p25 = K.reshape(percentil(img, 25.0), (nbatch, 1, 1, 3))
    p02 = K.reshape(percentil(img, 2.5), (nbatch, 1, 1, 3))
    iqr = tf.math.subtract(p75, p25)
    vmax_c = K.clip(tf.math.add(p75, tf.math.multiply(n15,iqr)), 0., 1.0)
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmin_c = tf.math.exp(-tf.math.sqrt(tf.math.multiply(p02, vmin)))
    vmin_c = tf.math.multiply(vmin_c, p02)
    imgw = K.clip(img, vmin_c, vmax_c)
    
    return imgw


def valsr_(img):
    """Vals sr"""

    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    bw = K.clip((vmax - vmin), 0.001, 1.0)
    
    return vmax, vmin, bw


def cena_(img):
    """Cena"""

    vmax, vmin, bw, vmed, B = vals(img)
    ic = tf.math.subtract(img, vmin)
    ic = K.clip(tf.math.divide_no_nan(ic, bw), 0., 1.0)
    
    return ic
    

def haze_(img):
    """Haze"""

    vmax, vmin, bw, vmed, B = vals(img)    
    # vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    # vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    hz = K.clip(tf.math.add(tf.math.multiply(img, bw), vmin), 0., 1.0)

    return hz

def param(img):
    """Param"""

    img = bw_adj(img)
    nbatch = K.shape(img)[0]
    Boo = K.reshape(get_median(img), (nbatch, 1, 1, 3))
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    #vmed = tf.reduce_mean(img, axis=(1,2), keepdims=True)
    bw = tf.math.multiply(vmax, vmin)
    gd = tf.math.divide_no_nan((1.0), bw)
    #gdi = tf.math.multiply(gd, (vmax-img+Boo)*img)
    gdi = tf.math.multiply(gd, (img-vmin)*(vmax - img))
    #gdi = tf.math.multiply(gdi, (1.0 - img))
    exp = tf.math.exp(-gdi)
    #gdb = tf.math.multiply(gd, (vmax-img+Boo)*Boo)
    gdb = tf.math.multiply(gd, (Boo-vmin)*(vmax - img))
    #gdb = tf.math.multiply(gdb, (img-Boo))
    expb = tf.math.exp(-gdb)
    
    return exp, expb, Boo


def fds(vmax, vmin, vmed):
    """fds function"""

    z5 = tf.constant([0.5], dtype=float)
    um = tf.constant([1.001], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    fmed = K.clip(tf.math.divide(K.abs(tf.math.subtract(z5, vmed)), dois), 0., 0.99)
    fs = tf.math.add(tf.math.subtract(um, vmax), fmed)
    fi = tf.math.add(vmin, fmed)
    fk = tf.math.divide(tf.math.multiply(fs, fi), tf.math.add(fs, fi))
    
    return fk


def param2_(img):
    """param2 function"""

    vmax, vmin, bw, vmed, B = vals(img)

    um = tf.constant([1.001], dtype=float)
    
    gbw = tf.math.divide_no_nan(tf.math.subtract(um, bw), bw)    
    gdi = tf.math.multiply(tf.math.subtract(vmax, img), tf.math.subtract(img, vmin))
    gdi = tf.math.multiply(gdi, gbw)

    gdb = tf.math.sqrt(tf.math.multiply(gdi, tf.math.subtract(B, vmin)))
    gdb = tf.math.multiply(gdb, gbw)

    exp = tf.math.exp(-gdi)
    expb = tf.math.exp(-gdb)

    return exp, expb, B 


def param_comp_(img):
    """param_comp"""

    #nbatch = K.shape(img)[0]
    #vmdn = K.reshape(get_median(img), (nbatch, 1, 1, 3))
    #vmed = tf.reduce_mean(img, axis=(1,2), keepdims=True)
    vmax, vmin, bw, vmed, B = vals(img)
    um = tf.constant([1.001], dtype=float)
    
    gbw = tf.math.divide_no_nan(tf.math.subtract(um, bw), bw)
    gd = tf.math.multiply(tf.math.subtract(vmax, img), tf.math.subtract(img, vmin))    
    gdb = tf.math.sqrt(tf.math.multiply(gd, tf.math.subtract(B, vmin)))
    #gdb = tf.math.sqrt(tf.math.add(gdb, 0.001))  # <-- raiz    
    gdb = tf.math.multiply(gdb, gbw)
    expb = tf.math.exp(-gdb)

    return expb, B


def tfg_(img, wd, sd):
    """TFG"""

    ig = K.abs(gaussian(img, [wd,wd], [sd,sd]))
    imggaus = K.clip(tf.math.subtract(tf.math.multiply(2.0, img), ig), 0., 1.0)
    
    return imggaus


def gauss_(img, wd, sd):
    """Gauss"""

    imgauss = K.clip(K.abs(gaussian(img, [wd,wd], [sd,sd])), 0., 1.0)
    
    return imgauss
    
# ============================================================================
# Ajuste da faixa dinamica- Consistencia temporal   

# =============================================================================
# Ajuste baseado no em average pooling
def exp_con_(ix, vn, vx):
    """Exp_con"""

    bwx = tf.math.subtract(vx, vn)
    ixc = tf.math.divide(tf.math.subtract(ix, vn), bwx)
    ixh = tf.math.add(tf.math.multiply(ix, bwx), vn)
    
    return ixc, ixh


def sig(img):
    """Sig"""

    um = tf.constant([1.0], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    m45 = tf.constant([-4.5], dtype=float)
    z5 = tf.constant([0.5], dtype=float)    
    fsig = tf.math.divide(um, tf.math.add(um, tf.math.exp(tf.math.multiply(m45, tf.math.subtract(img, z5)))))
    img = tf.math.divide(tf.math.add(img, fsig), dois)
    return img


def contemp3(imgg):
    """Contemp3"""

    #um = tf.constant([1.0], dtype=float)    
    dois = tf.constant([2.0], dtype=float)
    #tres = tf.constant([3.0], dtype=float)
    #quatro = tf.constant([4.0], dtype=float)    
    #m1 = tf.constant([-1.0], dtype=float)    
    cem = tf.constant([100.], dtype=float)
    dez = tf.constant([10.], dtype=float)
    nbatch = K.shape(imgg)[0]

    #imgg = sig(igg)
    #p = tf.constant([99.91, 99.92, 99.93, 99.94, 99.95, 99.96, 99.97, 99.98, 99.99, 100.], dtype=float)
    #p = tf.constant([99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9, 99.99], dtype=float)
    #p = tf.constant([98.5, 98,66, 98.83, 99, 99.17, 99.33, 99.5, 99.67, 99.83, 99.99], dtype=float)    
    p = tf.constant([97., 97.3, 97.7, 98., 98.3, 98.7, 99., 99.3, 99.66, 99.99], dtype=float)
    #p = tf.constant([98.65, 98,8, 98.95, 99.1, 99.25, 99.4, 99.55, 99.7, 99.85, 99.99], dtype=float)
    #p = tf.constant([95.5, 96, 96.5, 97, 97.5, 98, 98.5, 99, 99.5, 99.99], dtype=float)
    
    p0a = K.reshape(percentil(imgg, cem-p[0]), (nbatch, 1, 1, 3))
    p0b = K.reshape(percentil(imgg, cem-p[1]), (nbatch, 1, 1, 3))
    p0c = K.reshape(percentil(imgg, cem-p[2]), (nbatch, 1, 1, 3))
    p0d = K.reshape(percentil(imgg, cem-p[3]), (nbatch, 1, 1, 3))
    p0e = K.reshape(percentil(imgg, cem-p[4]), (nbatch, 1, 1, 3)) 
    p0f = K.reshape(percentil(imgg, cem-p[5]), (nbatch, 1, 1, 3))
    p0g = K.reshape(percentil(imgg, cem-p[6]), (nbatch, 1, 1, 3))
    p0h = K.reshape(percentil(imgg, cem-p[7]), (nbatch, 1, 1, 3))
    p0i = K.reshape(percentil(imgg, cem-p[8]), (nbatch, 1, 1, 3))
    p0j = K.reshape(percentil(imgg, cem-p[9]), (nbatch, 1, 1, 3))
    
    p99a = K.reshape(percentil(imgg, p[0]), (nbatch, 1, 1, 3))
    p99b = K.reshape(percentil(imgg, p[1]), (nbatch, 1, 1, 3)) 
    p99c = K.reshape(percentil(imgg, p[2]), (nbatch, 1, 1, 3))
    p99d = K.reshape(percentil(imgg, p[3]), (nbatch, 1, 1, 3))
    p99e = K.reshape(percentil(imgg, p[4]), (nbatch, 1, 1, 3))
    p99f = K.reshape(percentil(imgg, p[5]), (nbatch, 1, 1, 3))
    p99g = K.reshape(percentil(imgg, p[6]), (nbatch, 1, 1, 3))
    p99h = K.reshape(percentil(imgg, p[7]), (nbatch, 1, 1, 3))
    p99i = K.reshape(percentil(imgg, p[8]), (nbatch, 1, 1, 3))
    p99j = K.reshape(percentil(imgg, p[9]), (nbatch, 1, 1, 3))

    ima = K.clip(imgg, p0a, p99a)
    imb = K.clip(imgg, p0b, p99b)
    imc = K.clip(imgg, p0c, p99c)
    imd = K.clip(imgg, p0d, p99d)
    ime = K.clip(imgg, p0e, p99e)
    imf = K.clip(imgg, p0f, p99f)
    img = K.clip(imgg, p0g, p99g)
    imh = K.clip(imgg, p0h, p99h)
    imi = K.clip(imgg, p0i, p99i)
    imj = K.clip(imgg, p0j, p99j)
        
    imx1 = tf.math.add(tf.math.add(tf.math.add(tf.math.add(ima, imb), imc), imd), ime)
    imx2 = tf.math.add(tf.math.add(tf.math.add(tf.math.add(imf, img), imh), imi), imj)
    imx = tf.math.divide(tf.math.add(imx1, imx2), dez)  ##Fa

        
    icw = cena(imx)  #tf.math.divide(tf.math.add(cena(iw), icc), dois)
    ihw = haze(imx)  #tf.math.divide(tf.math.add(haze(iw), ihh), dois)
    iref = tf.math.divide(tf.math.add(icw, ihw), dois)  ##Fref
    ic = K.clip(tf.math.divide(tf.math.add(icw, iref), dois), 0., 1.0) 
    ih = K.clip(tf.math.divide(tf.math.add(ihw, iref), dois), 0., 1.0) 

    return iref, ic, ih  #, imgch    #, imgh, imgl


def ic_idb2_v1(img):
    """IC_idb2"""
    
    um = tf.constant([1.0], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    #tres = tf.constant([3.0], dtype=float)
    #imgw, ic, idb, imgh, imgl = contemp3(img)
    imgw, ic, idb = contemp3(img)
    exp, expb, B = param2(imgw)
    vmax, vmin, bw = valsr(imgw)
    Bmax = um
    ic = tf.math.subtract(ic, tf.math.multiply(B, tf.math.subtract(Bmax, expb)))
    ic = tf.math.divide_no_nan(ic, exp)
    #ic = adj_fx(ic)  ### <<<--- Equalizacao - fx dinamica
    ic = tfg(ic, 5, 3)
    #ic = tf.math.pow(ic, tf.constant([0.9], dtype=float))
    
    #ih = tf.math.add(tf.math.multiply(img_deg, bw), vmin)
    b0 = tf.math.multiply(idb, exp)
    b1 = tf.math.multiply(tf.math.subtract(Bmax, expb), B) 
    idb = tf.math.add(b0, b1)
    idb = tf.math.divide(tf.math.add(idb, gauss(idb, 5, 3)), dois)

    return ic, idb


def comp_(img, teta):
    """Comp"""

    expb, Boo = param_comp(img)
    compB = tf.math.multiply(teta, tf.math.multiply((1.0 - expb), Boo))
    compJ = tf.math.subtract(img, compB)
    
    return compJ, compB


def adj_fx(img):
    """Adj fx"""

    #um = tf.constant([1.0], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    #vmax, vmin, bw, vmed, B = vals(img)
    vmax, vmin, bw = valsr(img)
    #vmed = tf.math.reduce_mean(img, axis=(1,2), keepdims=True)
    fk = dois   #tf.math.add(um, tf.math.add(vmax, vmed))
    #fk = tf.math.add(um, tf.math.reduce_sum(vmed, axis=None, keepdims=False))
    cfd = tf.math.divide(tf.math.add(vmax, vmin), dois)
    i_adj = tf.math.multiply(img, tf.math.divide(tf.math.add(fk,cfd),tf.math.add(fk,img)))
    
    return i_adj


def udj_fx(img):
    """UDJ fx"""

    um = tf.constant([1.0], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    vmax, vmin, bw, vmed, B = vals(img)
    fk = tf.math.add(um, tf.math.add(vmax, vmed))
    cfd = tf.math.divide(tf.math.add(vmax, vmin), dois)
    i_udj = tf.math.multiply(img, tf.math.divide(tf.math.add(fk,img),tf.math.add(fk,cfd)))
    
    return i_udj    


def dblock_v1(img0, img1):
    """degradation block"""

    imgc, imgdb = ic_idb2(img0)
    img_map = tf.math.subtract(imgdb, imgc)
    img_out = K.clip(tf.math.add(img1, img_map), 0., 1)
    #imgh = toph(imgc)
    #img_out = tf.math.add(img_out, imgh)
    
    return img_out


def loss_YUVRGB(y_true, y_pred):
    """Loss fn"""

    eta = 0.       ###   <== fator de reducao da luminosidade de contexto na loss
    teta = tf.constant([1.0 - eta], dtype=float)    
      
    ka = tf.constant([0.6], dtype=float)      
    kb = tf.constant([0.4], dtype=float)  
    
    y_pred = dblock_v1(y_true, y_pred)
    compJt, compBt = comp(y_true, teta)
    compJp, compBp = comp(y_pred, teta)

    loss_mse_rgb = K.mean(K.square(y_true - y_pred))
    loss_compJ_rgb = K.mean(K.square(compJt - compJp))
    lossp = ka*loss_compJ_rgb + kb*loss_mse_rgb

    return lossp


def faz_DB(img):
    """Faz DB"""

    img1 = dblock_v1(img[0], img[1])
    
    return img1


class new_callback(tf.keras.callbacks.Callback):
    """Early stopping function"""

    def on_epoch_end(self, epoch, logs={}):
        lim = 50e-4
        if(logs.get('val_loss') < lim):
            rede_cDB.stop_training = True
            print("\n Loss de validacao menor que", lim, " - fim do treinamento") 
        return

callbacks = new_callback()


outputDB_img = Lambda(faz_DB, name='SDB')([input_img, output_img])

rede_cDB = Model(input_img, output_img, name='rede')
rede_cDB.compile(optimizer=opt, loss=loss_YUVRGB)

DB = Model(input_img, outputDB_img)
DB.compile(optimizer=opt, loss=loss_YUVRGB)

# ===========================================================================

# ===========================================================================

print("")
print("Treinando a rede com UWData_video:")
print('Epocas = ', epochs, ' e Batch-size = ', batch_size)

H0 = rede_cDB.fit(data, data,      
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=[callbacks],
              validation_split=0.1,
              steps_per_epoch= None,
              validation_steps=None)

saida = rede_cDB.predict(seleta)
saida_DB = DB.predict(seleta)


wall_time = datetime.now()
end_time = wall_time.strftime("%H:%M:%S")
print("Begin time: ", begin_time)
print("End time: ", end_time)


#esp_saida_rede_cDB2 = banda(esp_saida_rede_cDB)

#Salva as imagens de teste
def write_model():
    """Persist Model"""

    model_name = os.path.join(model_dir, 'Mod11Y_pre_videoB')
    rede_cDB.save(model_dir)
    model_name = os.path.join(model_dir, 'Mod11Y_pre_videoB_DB')
    DB.save('Mod11Y_pre_videoB_DB')


def  save_img(img, path, img_name):
    """
    Save image on directory
    
    An example
    
    ```python
        save_img(img, '/my/outputs/', 'image0')
    ```
    
    Args:
        img: image
        path: Absolute path
        img_name: name to save image
    
    Returns:
        None
    """
    img = img[:,:,::-1]
    img = np.uint8(255*img)

    path_dir = os.path.join(path, img_name)
    with open(img_name, 'wb') as f:
        pickle.dump(img, f)


qx = -1
plt.style.use('default')
for i in range(1, 5):

    plt.subplot(3, 4, i)
    p1 = plt.imshow(data[i+qx])
    plt.title('Entrada', fontsize=8)
    plt.axis('off')
    p1.axes.get_xaxis().set_visible(False)
    p1.axes.get_yaxis().set_visible(False)

    plt.subplot(3, 4, i+4)
    p2 = plt.imshow(saida[i+qx])
    plt.title('Saida rede', fontsize=8)    
    plt.axis('off')
    p2.axes.get_xaxis().set_visible(False)
    p2.axes.get_yaxis().set_visible(False)

    plt.subplot(3, 4, i+8)
    p3 = plt.imshow(saida_DB[i+qx])
    plt.title('Saida do DB', fontsize=8)    
    plt.axis('off')
    p3.axes.get_xaxis().set_visible(False)
    p3.axes.get_yaxis().set_visible(False)



