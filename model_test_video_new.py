#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:25:36 2021
by Claudio D. Mello Jr
Codigo para testar modelo previamente gravado
loss e demais funcoes sao carregadas a partir de "funcs_modelo_test"
"""

import tensorflow as tf

import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import numpy as np
import os

# Model functions
from funcs_model_test_video2 import dblock, loss_YUVRGB


def  salva_img(img, caminho, nome_img):
    """save image on hd"""

    img = img[:,:,::-1]
    img = np.uint8(255*img)
    with open(nome_img, 'wb') as f:
        pickle.dump(img, f)
    

def escala2(file):
    """escale 2"""

    new_file = []
    for i in range(len(file)):
       vmin = np.amin(file[i], axis=(0,1))
       vmax = np.amax(file[i], axis=(0,1))
       bw  = vmax - vmin
       img_c = 1 - (vmax - file[i])/bw
       new_file.append(img_c)

    return new_file


def quad(chn):
       """quad function"""

       vmin = np.amin(chn)
       vmax = np.amax(chn)
       bw = vmax - vmin
       a = (1 + bw*vmax)/bw
       
       return a, vmin

       
def escala3(file):
    """escale 3"""
    
    new_file = []
    for i in range(len(file)):
       vmin = np.amin(file[i], axis=(0,1))
       #vmax = np.amax(file[i], axis=(0,1))

       if(vmin[0] < 0): file[i][:,:,0] = (file[i][:,:,0] + np.abs(vmin[0]))
       if(vmin[1] < 0): file[i][:,:,1] = (file[i][:,:,1] + np.abs(vmin[1]))
       if(vmin[2] < 0): file[i][:,:,2] = (file[i][:,:,2] + np.abs(vmin[2]))

       vmax0 = np.amax(file[i], axis=(0,1))
       file[i][:,:,0]=file[i][:,:,0]/vmax0[0]
       file[i][:,:,1]=file[i][:,:,1]/vmax0[1]
       file[i][:,:,2]=file[i][:,:,2]/vmax0[2]
       if(vmax0[0] > 1): file[i][:,:,0] = (1 + file[i][:,:,0]*(1 - file[i][:,:,0]))*file[i][:,:,0]
       if(vmax0[1] > 1): file[i][:,:,1] = (1 + file[i][:,:,1]*(1 - file[i][:,:,1]))*file[i][:,:,1]
       if(vmax0[2] > 1): file[i][:,:,2] = (1 + file[i][:,:,2]*(1 - file[i][:,:,2]))*file[i][:,:,2]

       new_file.append(file[i])

    return new_file


def escala4(file):
    """escale 4"""

    new_file = []
    for i in range(len(file)):
       vmin = np.amin(file[i], axis=(0,1))
       #vmax = np.amax(file[i], axis=(0,1))

       if(vmin[0] < 0): 
           ar, vminr = quad(file[i][:,:,0])
           file[i][:,:,0] = (file[i][:,:,0] - vminr)*(ar - file[i][:,:,0])
           
       if(vmin[1] < 0): 
           ag, vming = quad(file[i][:,:,1])
           file[i][:,:,1] = (file[i][:,:,1] - vming)*(ag - file[i][:,:,1])
           
       if(vmin[2] < 0):
           ab, vminb = quad(file[i][:,:,2])
           file[i][:,:,2] = (file[i][:,:,2] - vminb)*(ab - file[i][:,:,2])

       new_file.append(file[i])

    return new_file            

def escala5(img):
    """escale 5"""

    vmin = np.amin(img, axis=(0,1))
    vmax = np.amax(img, axis=(0,1))
    vm = np.clip(((-1)*vmin), 0., 1.0)
    #vm = K.clip(vm, 0., 0.3)
    imgx = vm + img
    nmax = np.amax(imgx, axis=(0,1))
    nmax = np.clip(nmax, 1.0, vmax)
    imgx = imgx/nmax

    return imgx

# Directorys
#os.chdir(/tf/app/autoencoder)
wdir = os.getcwd()
work_dir = wdir
output_dir = os.path.join(work_dir, "outputs")
#
img_dataset_dir = os.path.join(work_dir,"datasets/images")
validation_image_dataset_dir = os.path.join(work_dir,"datasets/images")
#
video_dataset_dir = os.path.join(work_dir,"datasets/videos")
raw_video_dir = ""
#
model_dir = os.path.join(work_dir, "models/checkpoints")

model_name = 'Mod11Y_new_gdb'

#================ Load images ===========================
dataset_name = 'UWTest256X'
with open(dataset_name, 'rb') as f:
    avalia = pickle.load(f)

avalia = np.clip(avalia, 0., 1.0)


# ========================= Load model ===================================
modelo = tf.keras.models.load_model(os.path.join(model_dir,model_name), custom_objects = {                                                                             'dblock':dblock,
                                                                                      'loss_YUVRGB':loss_YUVRGB})

#modeloDB = tf.keras.models.load_model(diretorio_modelo11Y + modelo_idDB, custom_objects = {'loss_imag':loss_imag,                                                                                   'dblock':dblock,
#                                                                                      'loss_YUVRGB':loss_YUVRGB})
# ========================= Faz a predicao das imagens de saida ==============
#pred_modelo = modelo.predict(seleta)
# -------------------------------------------
walltime = datetime.now()
begin_time = walltime.strftime("%H:%M:%S.%f")[:-2]
pred_modelo = np.clip(modelo.predict(avalia), 0., 1.0)


def grava_file(file, name, path):
    """save output images
    
    Args:
        file: output image
        name: filename
        path: absolute path
    """
    name = path + '/' + name

    with open(name, 'wb') as f:
        pickle.dump(file, f)
    
    return
# ============================================================================
    
# ================== Mostra imagens preditas pelo modelo sob teste ===========
qx = -1
plt.style.use('default')
for i in range(1, 5):
    
    plt.subplot(2, 4, i)
    p1 = plt.imshow(avalia[i+qx])
    plt.title('Entrada', fontsize=8)
    plt.axis('off')
    p1.axes.get_xaxis().set_visible(False)
    p1.axes.get_yaxis().set_visible(False)
    
    plt.subplot(2, 4, i+4)
    p2 = plt.imshow(pred_modelo[i+qx])
    plt.title('Saida rede', fontsize=8)    
    plt.axis('off')
    p2.axes.get_xaxis().set_visible(False)
    p2.axes.get_yaxis().set_visible(False)



def treina(data, model, epochs, batch):
    """Fine-Tunning model
    
    Args:
        data: dataset. In format: [1,256,256,3]
        model: Keras model object: tf.keras.models.load_model()
        epochs: Fine-tunning epochs
        batch:
    """

    class new_callback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epochs, logs={}):
            lim = 30e-4
            if(logs.get('val_loss') < lim):    
                model.stop_training = True
                print("\nValidation loss less than", lim, " - FINISHING") 
                return
    
    callbacks = new_callback()
    
    print("Training")
    print('Epochs = ', epochs, ' e Batch-size = ', batch)

    #data = data**gamma

    model.fit(data, data,
              batch_size=batch,
              epochs=epochs,
              verbose=1,
              callbacks=[callbacks],
              validation_split=0.1,
              steps_per_epoch= None,
              validation_steps=None)


    print('\nInferindo...')
    output = model.predict(data)
    #saida_DB = modeldb.predict(data)
    
    return output
    
