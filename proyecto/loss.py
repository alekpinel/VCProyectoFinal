#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:22:51 2021

@author: Alejandro Pinel Martínez y Ángel de la Vega Jiménez
"""

import numpy as np
import proyectovc
import tensorflow as tf

def countClasses(masks):
    
    total_counts = {0:0, 1:0, 2:0}
    
    for mask in masks:
        classes = tf.argmax(mask, axis=-1).numpy()
        class_counts = np.unique(classes, return_counts=True)
        for c in range(len(class_counts[0])): 
            total_counts[class_counts[0][c]] += class_counts[1][c]
        
    return total_counts


def classWeights(class_weights):
    
    total = sum(class_weights.values())
    for key, value in class_weights.items():
        class_weights[key] =  total/value
    
    return class_weights


def calculateLossWeights(masks):
    
    mask_width = masks.shape[2]
    mask_height = masks.shape[1]
    class_count = countClasses(masks)
    class_weights = classWeights(class_count)
    w = [[class_weights[0], class_weights[1], class_weights[2]]] * mask_width
    h = [w] * mask_height
    
    return np.array(h)
    
    
    
''' ASÍ SE USA ESTO'''
X_train, y_train, X_test, y_test = proyectovc.LoadData()

prueba = calculateLossWeights(y_train)

''' Y luego en model.compile se le mete el parámetro
loss_weights = prueba, usando la métrica de accuracy normal 
'''