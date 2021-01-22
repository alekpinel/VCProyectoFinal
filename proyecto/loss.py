#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 12:22:51 2021

@author: Alejandro Pinel Martínez y Ángel de la Vega Jiménez
"""

import numpy as np

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
    
    print(class_weights)
    
    # Replicamos los pesos al tamaño de la máscara original
    return np.ones((mask_height, mask_width, 3))*class_weights