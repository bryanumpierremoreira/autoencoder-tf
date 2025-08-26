#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:43:40 2021

@author: nautec
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from fgaussian import gaussian
#from tensorflow.keras.layers import AveragePooling2D

# =========================== Calcula a mediana ==============================
def get_median(ix):
    """
    Get median from ix
    
    An example
    
    ```python
        get_median(img, '/my/outputs/', 'image0')
    ```
    
    Args:
        ix: tensor image
    
    Returns:
        Tensor
    """

    median = tfp.stats.percentile(ix, 50.0, axis=(1,2),
                                  interpolation='midpoint',
                                  preserve_gradients=True)
    # median = K.reshape(median, (nbatch, 1, 1, 3))
    return median

def get_median2(ix):
    md = tfp.stats.percentile(ix, 50.0, axis=None,
                                  interpolation='midpoint',
                                  preserve_gradients=True)
    # median = K.reshape(median, (nbatch, 1, 1, 3))
    return md
# ============================================================================
# ========================= Percentis ========================================
def percentil(img, per):
    """
    Percentil
    """
    perc = tfp.stats.percentile(img, per, axis=(1,2),
                                  interpolation='midpoint',
                                  preserve_gradients=True)
    return perc
# ============================================================================
def vals(img):
    """
    Image val
    """

    #nbatch = K.shape(img)[0]
    #vmdn = K.reshape(get_median(img), (nbatch, 1, 1, 3))
    dois = tf.constant([2.0], dtype=float)
    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    vmed = tf.reduce_mean(img, axis=(1,2), keepdims=True)
    B = tf.math.divide(tf.math.add(vmax, vmin), dois)    
    bw = K.clip((vmax - vmin), 0.001, 1.0)
    
    return vmax, vmin, bw, vmed, B

def valsr(img):
    """
    Vals sr
    
    """

    vmin = tf.reduce_min(img, axis=(1,2), keepdims=True)
    vmax = tf.reduce_max(img, axis=(1,2), keepdims=True)
    bw = K.clip((vmax - vmin), 0.001, 1.0)
    
    return vmax, vmin, bw

def exp_con(ix, vn, vx):
    """
    Exp_con
    """

    bwx = tf.math.subtract(vx, vn)
    ixc = tf.math.divide(tf.math.subtract(ix, vn), bwx)
    ixh = tf.math.add(tf.math.multiply(ix, bwx), vn)
    
    return ixc, ixh


def cena(img):
    """
    Cena
    """
    
    vmax, vmin, bw, vmed, B = vals(img)
    ic = tf.math.subtract(img, vmin)
    ic = K.clip(tf.math.divide_no_nan(ic, bw), 0., 1.0)
    
    return ic


def haze(img):
    """Haze"""

    vmax, vmin, bw, vmed, B = vals(img)    
    hz = K.clip(tf.math.add(tf.math.multiply(img, bw), vmin), 0., 1.0)       
    
    return hz

# ============ Contemp3 ======================================================
def contemp3(imgg):
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

    return iref, ic, ih  
    
# =============================================================================
# ========================= HSt e HSc baseadas na media =======================
def ted(img):
    um = tf.constant([1.0], dtype=float)
    vmax, vmin, B = valsr(img)
    vmed = tf.reduce_mean(img, axis=(1,2), keepdims=True)
    vn = tf.math.subtract(vmed, tf.math.multiply(vmed, tf.math.exp(-vmin)))
    vx = tf.math.add(vmed, tf.math.multiply((um-vmed), tf.math.exp(vmax-um)))
    bwn = tf.math.subtract(vx, vn)
    iwc = tf.math.divide(tf.math.subtract(img, vn), bwn)
    iwh = tf.math.add(tf.math.multiply(img, bwn), vn)
    return iwc, iwh
# =============================================================================


# ============ Ajuste sigmoidal  =============================================
def sig(img):
    um = tf.constant([1.0], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    m5 = tf.constant([-5.0], dtype=float)
    z5 = tf.constant([0.5], dtype=float)    
    fsig = tf.math.divide(um, tf.math.add(um, tf.math.exp(tf.math.multiply(m5, tf.math.subtract(img, z5)))))
    img = tf.math.divide(tf.math.add(img, fsig), dois)
    return img

# ============= Ajuste em relacao ao centro da faixa (Videos EMA) =============
def adj_fx(img):
    um = tf.constant([1.0], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    vmax, vmin, bw, vmed, B = vals(img)
    fk = tf.math.add(um, tf.math.add(vmax, vmed))
    cfd = tf.math.divide(tf.math.add(vmax, vmin), dois)
    i_adj = tf.math.multiply(img, tf.math.divide(tf.math.add(fk,cfd),tf.math.add(fk,img)))
    return i_adj

def udj_fx(img):
    um = tf.constant([1.0], dtype=float)
    dois = tf.constant([2.0], dtype=float)
    vmax, vmin, bw, vmed, B = vals(img)
    fk = tf.math.add(um, tf.math.add(vmax, vmed))
    cfd = tf.math.divide(tf.math.add(vmax, vmin), dois)
    i_udj = tf.math.multiply(img, tf.math.divide(tf.math.add(fk,img),tf.math.add(fk,cfd)))
    return i_udj
# =============================================================================
def param2(img):
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


def param_comp(img):
    """
    Param_comp
    """

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


def tfg(img, wd, sd):
    """
    TFG
    """

    ig = K.abs(gaussian(img, [wd,wd], [sd,sd]))
    imgg = K.clip(tf.math.subtract(tf.math.multiply(2.0, img), ig), 0., 1.0)
    
    return imgg


def gauss(img, wd, sd):
    """
    Gauss
    """

    imgauss = K.clip(K.abs(gaussian(img, [wd,wd], [sd,sd])), 0., 1.0)
    
    return imgauss


def ic_idb2_v2(img):
    """IC_idb2"""
    dois = tf.constant([2.0], dtype=float)
    tres = tf.constant([3.0], dtype=float)
    
    imgw, ic, idb = contemp3(img)
    exp, expb, B = param2(imgw)

    vmax, vmin, bw = valsr(imgw)
    
    # Bmax evita resultado negativo ou zero na imagem guia de enh em videos com valores muito baixos
    Bmax = tf.math.divide(tf.math.add(dois, vmax), tres)
    #Bmax = vmax
    #Bmax = tf.math.divide(tf.math.add(um, vmax), dois)
    ic = tf.math.subtract(ic, tf.math.multiply(B, tf.math.subtract(Bmax, expb)))
    ic = K.clip(tf.math.divide_no_nan(ic, exp), 0., 1.0)
    ic = adj_fx(ic)
    ic = tfg(ic, 3, 3)
    
    #ic = tf.math.pow(ic, tf.constant([0.9], dtype=float))
    
    #ih = tf.math.add(tf.math.multiply(img_deg, bw), vmin)
    b0 = tf.math.multiply(idb, exp)
    b1 = tf.math.multiply(tf.math.subtract(Bmax, expb), B) 
    idb = (tf.math.add(b0, b1))
    #idb = adj_fx(idb)
    idb = K.clip(tf.math.divide(tf.math.add(idb, gauss(idb, 3, 3)), dois), 0., 1.0)

    return ic, idb
# ============================================================================
# ============================ Componentes ===================================
def comp(img, teta):
    """Comp"""

    expb, Boo = param_comp(img)
    compB = tf.math.multiply(teta, tf.math.multiply((1.0 - expb), Boo))
    compJ = tf.math.subtract(img, compB)
    
    return compJ, compB


#============================ Bloco de degradacao ============================
def dblock_v2(img0, img1):
    """degradation block"""

    imgc, imgdb = ic_idb2(img0)
    img_map = tf.math.subtract(imgdb, imgc)
    img_out = K.clip(tf.math.add(img1, img_map), 0.001, 1.0)
    
    return img_out


# ================================== LOSS ====================================
def loss_YUVRGB(y_true, y_pred):
    """Loss fn"""

    eta = 0.       ###   <== fator de reducao da luminosidade de contexto na loss
    teta = tf.constant([1.0 - eta], dtype=float)    
      
    ka = tf.constant([0.60], dtype=float)      
    kb = tf.constant([0.40], dtype=float)  
 
    y_pred = dblock_v2(y_true, y_pred)
    compJt, compBt = comp(y_true, teta)
    compJp, compBp = comp(y_pred, teta)

    loss_mse_rgb = K.mean(K.square(y_true - y_pred))
    loss_compJ_rgb = K.mean(K.square(compJt - compJp))
    lossp = ka*loss_compJ_rgb + kb*loss_mse_rgb
    return lossp

# ============================================================================

def banda(img):
    """Banda function"""

    fx = 1.36/(0.9 + np.exp(-2.7*(img - 0.502))) - 0.31/(img + 9) - 0.18
    #fxs = np.array(fx, dtype=float)

    return fx

