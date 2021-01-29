#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:22:51 2021
@author: Alejandro Pinel Martínez y Ángel de la Vega Jiménez
"""

import numpy as np
import keras.backend as K
import tensorflow as tf


def calculateClassWeights(masks):
    '''
    Calculate a loss weight for each class inversely proportional to the
    occurences of each class
    '''    
    count_per_class = np.sum(masks, axis=(0,1,2)) # shape(num_classes)
    return np.max(count_per_class)/(count_per_class.astype(np.float32))


def weighted_categorical_crossentropy(weights):
    '''
    Weighted categorical cross-entropy loss function
    '''
    def loss(y_true, y_pred):
        
        # Based in the following webs, but errors corrected
        # https://stackoverflow.com/questions/59609829/weighted-pixel-wise-categorical-cross-entropy-for-semantic-segmentation
        # https://stackoverflow.com/questions/59520807/multi-class-weighted-loss-for-semantic-image-segmentation-in-keras-tensorflow
        

        # Categorical crossentropy for prediction on each pixel
        unweighted_loss = K.categorical_crossentropy(y_true, y_pred) # shape(batchSize, 256, 256)
        
        # Weights for each pixel based on its class
        yWeights = weights * y_true # shape(batchSize, 256, 256, 3)
        yWeights = K.sum(yWeights, axis=-1) # shape(batchSize, 256, 256)
        
        # Multiply loss on each pixel by the weight corresponding to the class
        weighted_loss = yWeights * unweighted_loss
                                        
        # Following links confirm that the output could be one value per pixel, or
        # a value per batch.
        # https://stackoverflow.com/questions/52034983/how-is-total-loss-calculated-over-multiple-classes-in-keras
        # https://stackoverflow.com/questions/63390725/should-the-custom-loss-function-in-keras-return-a-single-loss-value-for-the-batc
        # https://github.com/tensorflow/tensorflow/issues/42446
        
        return weighted_loss # shape (batchSize, 256, 256)

    return loss


def dice_loss(y_true, y_pred):
    '''
    Loss function based on Dice coefficient. Returns single value per image.
    '''
    
    axes = (1,2) # weight and height axis of images
    smooth = 0.001 # avoid zero division in case of not present class
    
    # Equivalent to the true positives on each class
    intersection = K.sum(K.abs(y_true*y_pred), axis = axes)
    
    # 2TP + FP + FN
    mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    
    dice = (2*intersection + smooth)/(mask_sum + smooth)
    
    # Ignore the unrepresented classes in the mean calculation
    mask = K.cast(K.not_equal(mask_sum, 0), 'float32')
    non_zero_count = K.sum(mask, axis = -1)
    non_zero_sum = K.sum(dice * mask, axis = -1)
    
    # Dice for each class 
    return 1 - non_zero_sum/non_zero_count #shape(batchSize,3)


# Based in the following code, simplified:
# https://gist.github.com/ilmonteux/8340df952722f3a1030a7d937e701b5a
def seg_metrics(y_true, y_pred, metric_name):
    '''
    Calculate two posible metrics for segmentation:
        - Mean dice coefficient for 3 classes
        - Mean inserseccion over union for 3 classes
    '''

    # intersection and union shapes are batch_size * n_classes (values = area in pixels)
    axes = (1,2) # W,H axes of each image
    intersection = K.sum(K.abs(y_true * y_pred), axis=axes) # Total TP for each class
    
    # 2TP + FP + FN, for each class (used for the dice coefficient)
    mask_sum = K.sum(K.abs(y_true), axis=axes) + K.sum(K.abs(y_pred), axis=axes)
    
    union = mask_sum  - intersection

    smooth = .001 # avoid zero division
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
    '''
    Returns the iuo metric function
    '''
    return seg_metrics(y_true, y_pred, metric_name='iou')


def mean_dice(y_true, y_pred):
    '''
    Returns the dice metric function
    '''
    return seg_metrics(y_true, y_pred, metric_name='dice')

