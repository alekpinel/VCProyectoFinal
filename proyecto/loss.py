#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:22:51 2021

@author: Alejandro Pinel Martínez y Ángel de la Vega Jiménez
"""

import numpy as np
import math
import keras
import keras.backend as K
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
    # return np.ones((mask_height, mask_width, 3))*class_weights
    return class_weights


def calculateClassWeights(masks):
    
    weights = {}
    
    count_per_class = np.sum(masks, axis=(0,1,2))
    num_classes = len(count_per_class)
    total = np.sum(count_per_class)
        
    for i, c in enumerate(count_per_class):
        weights[i] = total/(num_classes * c)
        
    return weights

def WeightedCategoricalCrossEntropy(weights):
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
    
    # weights = np.array(weights)
    # def loss(y_true, y_pred):
    #     # scale predictions so that the class probas of each sample sum to 1
    #     y_pred /= np.sum(y_pred, axis=-1, keepdims=True)
    #     # clip to prevent NaN's and Inf's
    #     y_pred = np.clip(y_pred, 0, 1)
    #     # calc
    #     loss = y_true * np.log(y_pred) * weights
    #     loss = -np.sum(loss, -1)
    #     return loss
    
    # return loss

