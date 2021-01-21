# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:05:22 2021

@author: Alejandro Pinel Martínez - Ángel De la Vega Jiménez
"""

datapath = "../data/" #Local
pretrainingdatapath = "../data_pretraining/"
savedmodelspath = "./saves/"
pretrainedUNet = savedmodelspath + "PretrainedUNet.h5"

import keras
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from keras.applications.resnet import ResNet50
from keras.utils import plot_model, to_categorical

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, AveragePooling2D
from keras.layers import Add, Concatenate
from keras.layers import UpSampling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from PIL import Image

from keras.optimizers import SGD

#The local GPU used to run out of memory, so we limited the memory usage:
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def ToGray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def LoadGif(filename):
    gif = cv2.VideoCapture(filename)
    ret,frame = gif.read()
    img = frame
    # img = Image.fromarray(frame)
    # img = img.convert('RGB')
    return img

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
    imagespath = datapath + "images/"
    maskspath = datapath + "masks/"
    
    names = os.listdir(imagespath)
    
    #Load data into lists
    listimages = []
    listmasks = []
    for imagename in tqdm(names):
        listimages.append(LoadImage(imagespath + imagename))
        listmasks.append(LoadImage(maskspath + imagename, False))
    
    listimages = [cv2.resize(img, target_size[::-1]) for img in listimages]
    listmasks  = [cv2.resize(img, target_size[::-1]) for img in listmasks]
    
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
    
    Y_train = ToCategoricalMatrix(Y_train)
    Y_test = ToCategoricalMatrix(Y_test)
    
    return X_train, Y_train, X_test, Y_test

def LoadPretrainingData(target_size=(256, 256)):
    trainpath = pretrainingdatapath + "training/"
    testpath = pretrainingdatapath + "test/"
    
    def LoadFolder(path, masks=False):
        data = []
        names = os.listdir(path)
        for imagename in tqdm(names):
            if (masks):
                img = LoadGif(path + imagename)
                img = ToGray(img)
                data.append(img)
            else:
                data.append(LoadImage(path + imagename, True))
        return data
    
    trainimages = LoadFolder(trainpath + 'images/')
    trainmasks = LoadFolder(trainpath + 'mask/', True)
    testimages = LoadFolder(testpath + 'images/')
    testmasks = LoadFolder(testpath + 'mask/', True)
    
    trainimages = [cv2.resize(img, target_size[::-1]) for img in trainimages]
    trainmasks = [cv2.resize(img, target_size[::-1]) for img in trainmasks]
    testimages = [cv2.resize(img, target_size[::-1]) for img in testimages]
    testmasks = [cv2.resize(img, target_size[::-1]) for img in testmasks]
    
    X_train = np.stack(trainimages)
    Y_train = np.stack(trainmasks)
    X_test = np.stack(testimages)
    Y_test = np.stack(testmasks)
    
    X_train = X_train / 255
    Y_train = Y_train / 255
    X_test = X_test / 255
    Y_test = Y_test / 255
    
    # Y_train = ToCategoricalMatrix(Y_train)
    # Y_test = ToCategoricalMatrix(Y_test)
    
    return X_train, Y_train, X_test, Y_test
    

def ToCategoricalMatrix(data):
    originalShape = data.shape
    totalFeatures = data.max() + 1
    
    categorical = data.reshape((-1,))
    categorical = to_categorical(categorical)
    data = categorical.reshape(originalShape + (totalFeatures,))
    return data

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


    
#UNet from a ResNet
def UNetFromResNet(input_shape=(256, 256, 3), n_classes=3):
    #This models a decoder block
    def DecoderBlock(filters, x, skip):
        
        x = UpSampling2D(size=2)(x)
        # print(skip.output_shape)
        x = Concatenate()([x, skip])
        
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        
        return x
    
    backbone = ResNet50(input_shape = input_shape, include_top = False, weights = 'imagenet', pooling = 'avg')
    model_input = backbone.input
    
    #We eliminate the last average pooling
    x = backbone.layers[-2].output
    
    
    #Layers were we are going to do skip connections.
    feature_layers = [142, 80, 38, 4, 0]
    filters = [1024, 512, 256, 64, 32]
    
    for i in range(len(feature_layers)):
        skip = backbone.layers[feature_layers[i]].output
        x = DecoderBlock(filters[i], x, skip)
    
    #Final Convolution
    x = Conv2D(n_classes, (3, 3), activation='sigmoid', padding='same')(x)
    
    model_output = x
    model = Model(model_input, model_output)
    
    return model

#Classic implementation of UNet
def UNetClassic(input_shape=(256, 256, 3), n_classes=3):
    #Layer of encoder: 2 convs and pooling
    def EncoderLayer(filters, x):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        feature_layer = x
        x = MaxPooling2D()(x)
        return x, feature_layer
    #Layer of decoder, upsampling, conv, concatenation and 2 convs
    def DecoderLayer(filters, x, skip):
        x = UpSampling2D(size=(2,2))(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Concatenate()([x, skip])
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        return x
    
    #Input
    x = Input(input_shape)
    model_input = x
    
    #Encoder
    x, encoder1 = EncoderLayer(64,  x)
    x, encoder2 = EncoderLayer(128, x)
    x, encoder3 = EncoderLayer(256, x)
    x, encoder4 = EncoderLayer(512, x)
    
    #Centre
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    
    #Decoder
    x = DecoderLayer(512, x, encoder4)
    x = DecoderLayer(256, x, encoder3)
    x = DecoderLayer(128, x, encoder2)
    x = DecoderLayer(64,  x, encoder1)
    
    #Output
    x = Conv2D(n_classes, (3, 3), activation='sigmoid', padding='same')(x)
    model_output = x
    
    model = Model(model_input, model_output)
    return model


# Compile with the optimizer and the loss function
def Compile(model, loss='categorical'):
    optimizer = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    
    if (loss == 'binary'):
        loss = keras.losses.binary_crossentropy
    else:
        loss = keras.losses.categorical_crossentropy
    
    
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

def Train(model, X_train, Y_train, X_val, Y_val, batch_size=128, epochs=12):
    hist = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_val, Y_val))
    return hist

def Test(model, X_test, Y_test):
    predicciones = model.predict(X_test)
    labels = np.argmax(Y_test, axis = 1)
    preds = np.argmax(predicciones, axis = 1)
    accuracy = sum(labels == preds)#/len(labels)
    return accuracy

def AdjustModel(model, n_classes):
    model_input = model.input
    
    #We eliminate the last layer
    x = model.layers[-2].output
    model_output = Conv2D(n_classes, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(model_input, model_output)
    
    return model
    
def PreTrain(model, pathtosave):
    X_train, Y_train, X_test, Y_test = LoadPretrainingData()
    model = AdjustModel(model, 1)
    Compile(model, loss='binary')
    Train(model, X_train, Y_train, X_test, Y_test, batch_size=4, epochs=30)
    model.save(pathtosave)

def LoadModel(pathtosave, n_classes):
    model = keras.models.load_model(pathtosave)
    model = AdjustModel(model, n_classes)
    return model

def main():
    # X_train, Y_train, X_test, Y_test = LoadData()
    
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    
    #Pretrain block
    # unet = UNetClassic()
    # PreTrain(unet, pretrainedUNet)
    
    X_train, Y_train, X_test, Y_test = LoadPretrainingData()
    model = LoadModel(pretrainedUNet, 1)
    Compile(model, loss='binary')
    Train(model, X_train, Y_train, X_test, Y_test, batch_size=4, epochs=30)
    
    # ClassPertentage(Y_train)
    
    # unet = UNetClassic()
    # unet.summary()
    # Compile(unet)
    # Train(unet, X_train, Y_train, X_test, Y_test, batch_size=1, epochs=5)
    
    # unet.save(savedmodelspath + 'UNet.h5')
    
    
    # model = keras.models.load_model(savedmodelspath + 'UNet.h5')
    # acc = Test(model, X_test, Y_test)
    # print(f"Accuracy is: {acc}")
    
    

if __name__ == '__main__':
  main()