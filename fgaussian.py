#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 08:17:34 2022

@author: nautec
"""
import numpy as np
import tensorflow as tf

def pad(input, ksize, mode, constant_values):
    input = tf.convert_to_tensor(input)
    ksize = tf.convert_to_tensor(ksize)
    mode = "CONSTANT" #if mode is None else upper(mode)
    constant_values = (
        tf.zeros([], dtype=input.dtype)
        if constant_values is None
        else tf.convert_to_tensor(constant_values, dtype=input.dtype)
    )

    assert mode in ("CONSTANT", "REFLECT", "SYMMETRIC")

    height, width = ksize[0], ksize[1]
    top = (height - 1) // 2
    bottom = height - 1 - top
    left = (width - 1) // 2
    right = width - 1 - left
    paddings = [[0, 0], [top, bottom], [left, right], [0, 0]]
    return tf.pad(input, paddings, mode=mode, constant_values=constant_values)


def gaussian(input, ksize, sigma, mode=None, constant_values=None, name=None):
    """
    Apply Gaussian filter to image.
    Args:
      input: A 4-D (`[N, H, W, C]`) Tensor.
      ksize: A scalar or 1-D `[kx, ky]` Tensor.
        Size of the Gaussian kernel.
        If scalar, then `ksize` will be broadcasted to 1-D `[kx, ky]`.
      sigma: A scalar or 1-D `[sx, sy]` Tensor.
        Standard deviation for Gaussian kernel.
        If scalar, then `sigma` will be broadcasted to 1-D `[sx, sy]`.
      mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
        (case-insensitive). Default "CONSTANT".
      constant_values: A `scalar`, the pad value to use in "CONSTANT"
        padding mode. Must be same type as input. Default 0.
      name: A name for the operation (optional).
    Returns:
      A 4-D (`[N, H, W, C]`) Tensor.
    """

    input = tf.convert_to_tensor(input)
    ksize = tf.convert_to_tensor(ksize)
    sigma = tf.cast(sigma, input.dtype)

    def kernel1d(ksize, sigma, dtype):
        x = tf.range(ksize, dtype=dtype)
        x = x - tf.cast(tf.math.floordiv(ksize, 2), dtype=dtype)
        x = x + tf.where(
            tf.math.equal(tf.math.mod(ksize, 2), 0), tf.cast(0.5, dtype), 0
        )
        g = tf.math.exp(-(tf.math.pow(x, 2) / (2 * tf.math.pow(sigma, 2))))
        g = g / tf.reduce_sum(g)
        return g

    def kernel2d(ksize, sigma, dtype):
        kernel_x = kernel1d(ksize[0], sigma[0], dtype)
        kernel_y = kernel1d(ksize[1], sigma[1], dtype)
        return tf.matmul(
            tf.expand_dims(kernel_x, axis=-1),
            tf.transpose(tf.expand_dims(kernel_y, axis=-1)),
        )

    ksize = tf.broadcast_to(ksize, [2])
    sigma = tf.broadcast_to(sigma, [2])
    g = kernel2d(ksize, sigma, input.dtype)

    input = pad(input, ksize, mode, constant_values)

    channel = tf.shape(input)[-1]
    shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
    g = tf.reshape(g, shape)
    shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
    g = tf.broadcast_to(g, shape)
    return tf.nn.depthwise_conv2d(input, g, [1, 1, 1, 1], padding="VALID")
