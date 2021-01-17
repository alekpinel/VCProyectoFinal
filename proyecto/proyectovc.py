# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:05:22 2021

@author: Alejandro Pinel Martínez - Ángel De la Vega Jiménez
"""

datapath = "../data/" #Local

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

#Function that load and image and convert it to RGB if needed
def LoadImage(filename, color = True):
    if (color):
      img = cv2.imread(filename,cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

    return img

#Shows an Image using Matplotlib 
def ShowImage(img, title=None):
    plt.imshow(img, cmap='gray')
    if (title != None):
        plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.show()

#Load the data from the datapath directory and resize all the images
def LoadData(testpercent = 0.2, target_size=(256, 256)):
    imagespath = datapath + "/images/"
    maskspath = datapath + "/masks/"
    
    names = os.listdir(imagespath)
    
    #Load data into lists
    listimages = []
    listmasks = []
    for imagename in tqdm(names):
        listimages.append(LoadImage(imagespath + imagename))
        listmasks.append(LoadImage(maskspath + imagename, False))
    
    listimages = [cv2.resize(img, target_size[::-1]) for img in listimages]
    listmasks = [cv2.resize(img, target_size[::-1]) for img in listmasks]
    
    #Randomize the order
    zipped = list(zip(listimages, listmasks))
    random.shuffle(zipped)
    listimages, listmasks = zip(*zipped)
    
    #Split training and test
    trainimages = listimages[:int(len(listimages)*(1 - testpercent))]
    testimages = listimages[int(len(listimages)*(1 - testpercent)):]
    
    trainmasks = listmasks[:int(len(listmasks)*(1 - testpercent))]
    testmasks = listmasks[int(len(listmasks)*(1 - testpercent)):]
    
    X_train = np.stack(trainimages)
    Y_train = np.stack(trainmasks)
    X_test = np.stack(testimages)
    Y_test = np.stack(testmasks)
    
    return X_train, Y_train, X_test, Y_test

#Show the percent of each class
def ClassPertentage(masks):
    unique, counts = np.unique(masks, return_counts=True)
    total = sum(counts)
    percents = [x/total*100 for x in counts]
    data = [("Background", percents[0], 'blue'), ("Blood cells", percents[1], 'red'), 
            ("Bacteries", percents[2], 'green')]
    print(f"Percent of pixels of each class:\nBackground: {percents[0]}\nBlood cells: {percents[1]}\nBacteries {percents[2]}")
    
    PlotBars(data, "Class Percentages", "Percent")

#Plot bars. Data must be ("title", value) or ("title", value, color)
def PlotBars(data, title=None, y_label=None):
    strings = [i[0] for i in data]
    x = [i for i in range(len(data))]
    y = [i[1] for i in data]
    
    colors=None
    if (len(data[0]) > 2):
        colors = [i[2] for i in data]
    
    fig, ax = plt.subplots()
    
    if (title is not None):
        ax.set_title(title)
    if (y_label is not None):
        ax.set_ylabel(y_label)
    
    # fig.autofmt_xdate()
    x_labels=strings
    plt.xticks(x, x_labels)
    
    if (colors is not None):
        plt.bar(x, y, color=colors)
    else:
        plt.bar(x, y)
    plt.show()

def main():
    X_train, Y_train, X_test, Y_test = LoadData()
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    ClassPertentage(Y_train)
    

if __name__ == '__main__':
  main()