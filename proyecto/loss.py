#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:22:51 2021

@author: Alejandro Pinel Martínez y Ángel de la Vega Jiménez
"""

import numpy as np
from keras import backend as K
import tensorflow as tf

def calculateLossWeights(masks):
    '''
    Calculate a loss weight for each class inversely proportional to the
    occurences of each class
    '''
    mask_width = masks.shape[2]
    mask_height = masks.shape[1]
    
    count_per_class = np.sum(masks, axis=(0,1,2))
    
    # El peso de cada clase es inversamente porporcional a la proporción
    # en la que aparezca
    class_weights = np.sum(count_per_class)/count_per_class.astype(np.float64)
    
    # Replicamos los pesos al tamaño de la máscara original
    return np.ones((mask_height, mask_width, 3))*class_weights


def calculateClassWeights(masks):
    '''
    Calculate a loss weight for each class inversely proportional to the
    occurences of each class
    '''
    
    count_per_class = np.sum(masks, axis=(0,1,2))
    
    # El peso de cada clase es inversamente porporcional a la proporción
    # en la que aparezca
    return np.sum(count_per_class)/count_per_class.astype(np.float32)


# =============================================================================
# Funciones de pérdida
# =============================================================================

# WCCE OK
def weighted_categorical_crossentropy(weights):
    '''
    Weighted categorical cross-entropy loss function
    '''
    def loss(y_true, y_pred):
        
        # Basado en los siguientes enlaces, pero corregido error del primero
        # https://stackoverflow.com/questions/59609829/weighted-pixel-wise-categorical-cross-entropy-for-semantic-segmentation
        # https://stackoverflow.com/questions/59520807/multi-class-weighted-loss-for-semantic-image-segmentation-in-keras-tensorflow
        
        # Podría ser mejor opción (pensar):
        # yWeights = weights * y_pred
        
        yWeights = weights * y_true # shape(batch, 256, 256, 3)
        yWeights = K.sum(yWeights, axis=-1) # shape(batch, 256, 256)
        
        unweighted_loss = K.categorical_crossentropy(y_true, y_pred) # shape(batch, 256, 256)
        
        weighted_loss = yWeights * unweighted_loss
                                
        # ¡LO QUE PONEN EN LA RESPUESTA ACEPTADA DE STACKOVERFLOW NO ES CORRECTO!
        # return K.sum(weighted_loss, axis=(1,2)) # shape (batch, )
        # return K.mean(weighted_loss, axis=(1,2)) # shape (batch, )
        
        # Esto sí lo es, la función de pérdida debe dar una salida POR CADA EJEMPLO,
        # no por cada batch!!
        # https://stackoverflow.com/questions/52034983/how-is-total-loss-calculated-over-multiple-classes-in-keras
        # https://stackoverflow.com/questions/63390725/should-the-custom-loss-function-in-keras-return-a-single-loss-value-for-the-batc
        # Issue confirmado, se confirma que debe dar una salida por cada ejemplo
        # https://github.com/tensorflow/tensorflow/issues/42446
        return weighted_loss # shape (batch, 256, 256)

    return loss

# Dice creo que OK
def dice_loss(y_true, y_pred):
    
    axes = (1,2) # W,H of each image
    smooth = 0.001 # avoid zero division in case of not present class
    
    # TP
    intersection = K.sum(K.abs(y_true*y_pred), axis = axes)
    
    # 2TP + FP + FN
    mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    
    # Negative because keras will try to minimize
    # Another option: 1 - dice
    dice = - (2*intersection + smooth)/(mask_sum + smooth)
    
    # Mean only the represented classes
    mask = K.cast(K.not_equal(mask_sum, 0), 'float32')
    non_zero_count = K.sum(mask, axis = -1)
    non_zero_sum = K.sum(dice * mask, axis = -1)
        
    # Creo que esto debería ser equivalente
    # return K.mean(non_zero_sum/non_zero_count)
    
    # Devuelve Dice medio para cada imagen shape(batch,) 
    return non_zero_sum/non_zero_count
    
''' Uso '''
# loss = weighted_categorical_crossentropy(weights)
# model.compile(..., loss=loss)

# Pruebas mías
# loss = weighted_categorical_crossentropy(np.array([1,1,1]).astype(np.float32))
# true = np.array([[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]]])[np.newaxis, :].astype(np.float32)
# pred = np.array([[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,1,0],[0,0,1]]])[np.newaxis, :].astype(np.float32)


''' METRICAS '''
# https://gist.github.com/ilmonteux/8340df952722f3a1030a7d937e701b5a
def seg_metrics(y_true, y_pred, metric_name):

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1,2) # W,H axes of each image
    intersection = K.sum(K.abs(y_true * y_pred), axis=axes)
        
    mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
        
    union = mask_sum  - intersection # or, np.logical_or(y_pred, y_true) for one-hot

    smooth = .001 # Evitar división por 0
    iou = (intersection + smooth) / (union + smooth)
    dice = (2*intersection + smooth)/(mask_sum + smooth)

    metric = {'iou': iou, 'dice': dice}[metric_name]

    # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
    mask =  K.cast(K.not_equal(union, 0), 'float32')
    
    # take mean only over non-absent classes
    class_count = K.sum(mask, axis=0)    
    non_zero = tf.greater(class_count, 0)
    
    non_zero_sum = tf.boolean_mask(K.sum(metric * mask, axis=0), non_zero)
    
    non_zero_count = tf.boolean_mask(class_count, non_zero)
    
    return K.mean(non_zero_sum / non_zero_count)

def mean_iou(y_true, y_pred):
    """
    Compute mean Intersection over Union of two segmentation masks, via Keras.
    """
    return seg_metrics(y_true, y_pred, metric_name='iou')

def mean_dice(y_true, y_pred):
    """
    Compute mean Dice coefficient of two segmentation masks.
    """
    return seg_metrics(y_true, y_pred, metric_name='dice')