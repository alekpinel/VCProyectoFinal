# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 11:05:22 2021

@author: Alejandro Pinel Martínez - Ángel De la Vega Jiménez
"""

datapath = "../data/" #Local
pretrainingdatapath = "../data_pretraining/"
savedmodelspath = "./saves/"
pretrainedUNet = savedmodelspath + "PretrainedUNet.h5"
pretrainedUNetv2 = savedmodelspath + "PretrainedUNetv2.h5"
savedUNet = savedmodelspath + "SavedUNet.h5"
savedUNetv2 = savedmodelspath + "SavedUNetv2.h5"
tempUNet = savedmodelspath + "TempUNet.h5"

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
from keras.preprocessing.image import ImageDataGenerator  

from keras.optimizers import SGD
from sklearn.model_selection import KFold

from loss import *
from visualization import *

#The local GPU used to run out of memory, so we limited the memory usage:
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#Transform to gray
def ToGray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Function that load a gif
def LoadGif(filename):
    gif = cv2.VideoCapture(filename)
    ret,frame = gif.read()
    img = frame
    return img

#Function that load an image and convert it to RGB if needed
def LoadImage(filename, color = True):
    if (color):
      img = cv2.imread(filename,cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

    return img

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
    
    X_train = X_train / 255
    X_test = X_test / 255
    
    Y_train = ToCategoricalMatrix(Y_train)
    Y_test = ToCategoricalMatrix(Y_test)
    
    return X_train, Y_train, X_test, Y_test

#Load the pretraining data
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
    
#Get the generators from raw data and the arguments to make them
def GetGenerators(X_train, Y_train, X_test, Y_test, validation_split=0.1, batch_size=128, data_augmentation=False, seed=None,
                  shift_range=0.1, rotation_range=5, flip=True, zoom_range=0.2):
    basic_generator_args = dict(
        validation_split=validation_split
    )
    
    data_augmentation_generator_args = dict(
        width_shift_range=shift_range,
        height_shift_range=shift_range,
        rotation_range=rotation_range,
        
        horizontal_flip=flip,
        vertical_flip=flip,
        zoom_range=zoom_range,
        validation_split=validation_split
    )
    
    test_args = dict()
    
    if (seed is None):
        seed = 1
    
    if (data_augmentation):
        data_generator_args = data_augmentation_generator_args
    else:
        data_generator_args = basic_generator_args
        
    train_gen = GenerateData(X_train, Y_train, data_generator_args, subset='training', batch_size=batch_size, seed=seed)
    val_gen   = GenerateData(X_train, Y_train, data_generator_args, subset='validation', batch_size=batch_size, seed=seed)
    test_gen  = GenerateData(X_test, Y_test, test_args, batch_size=batch_size, seed=seed)
    
    return train_gen, val_gen, test_gen, data_generator_args, test_args
    
# Given X and Y, create a generator with the same seed for them
def GenerateData(X, Y, generator_args=None, subset=None, batch_size=4, seed=None):
    if (seed is None):
        seed = 1
    if (generator_args is None):
        generator_args = dict()
    
    image_datagen = ImageDataGenerator(**generator_args)
    masks_datagen = ImageDataGenerator(**generator_args)
    
    image_generator = image_datagen.flow(
        X, subset=subset, batch_size=batch_size, seed=seed)
    
    masks_generator = masks_datagen.flow(
        Y, subset=subset, batch_size=batch_size, seed=seed)
    
    data_gen = zip(image_generator, masks_generator)
    return data_gen

#To_categorical for more than one dimension
def ToCategoricalMatrix(data):
    originalShape = data.shape
    totalFeatures = data.max() + 1
    
    categorical = data.reshape((-1,))
    categorical = to_categorical(categorical)
    data = categorical.reshape(originalShape + (totalFeatures,))
    return data

#A mask with one value for pixel between 0-2
def MaskMonoband(data):
    return np.argmax(data, axis=-1)

#Show the percent of each class
def ClassPercentage(masks):
    masks = MaskMonoband(masks)
    unique, counts = np.unique(masks, return_counts=True)
    total = sum(counts)
    percents = [x/total*100 for x in counts]
    data = [("Background", percents[0], 'blue'), ("Blood cells", percents[1], 'red'), 
            ("Bacteries", percents[2], 'green')]
    print(f"Percent of pixels of each class:\nBackground: {percents[0]}\nBlood cells: {percents[1]}\nBacteries {percents[2]}")
    
    PlotBars(data, "Class Percentages", "Percent", dateformat=False)

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
        # x = BatchNormalization()(x)
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
        # x = BatchNormalization()(x)
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
    if (n_classes == 1):
        x = Conv2D(n_classes, (1, 1), activation='sigmoid', padding='same')(x)
    else:
        x = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(x)
        
    model_output = x
    
    model = Model(model_input, model_output)
    return model

# Added BatchNormalization and dropout into classic Unet
def UNetV2(input_shape=(256, 256, 3), n_classes=3):
    #Layer of encoder: 2 convs and pooling
    def EncoderLayer(filters, x):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        feature_layer = x
        x = MaxPooling2D()(x)
        return x, feature_layer
    
    #Layer of decoder, upsampling, conv, concatenation and 2 convs
    def DecoderLayer(filters, x, skip):
        x = UpSampling2D(size=(2,2))(x)
        x = (BatchNormalization())(x)
        x = Concatenate()([x, skip])
        x = Conv2DTranspose(filters, (3, 3), activation='relu', padding='same')(x)
        x = (BatchNormalization())(x)
        x = Conv2DTranspose(filters, (3, 3), activation='relu', padding='same')(x)
        x = (BatchNormalization())(x)
        x = Conv2DTranspose(filters, (3, 3), activation='relu', padding='same')(x)
        x = (BatchNormalization())(x)
        x = Dropout(0.2)(x)
        return x
    
    #Input
    x = Input(input_shape)
    model_input = x
    
    #Encoder
    x, encoder1 = EncoderLayer(32,  x)
    x, encoder2 = EncoderLayer(64, x)
    x, encoder3 = EncoderLayer(64, x)
    x, encoder4 = EncoderLayer(128, x)
    x, encoder5 = EncoderLayer(256, x)
    
    #Centre
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    x = DecoderLayer(256, x, encoder5)
    x = DecoderLayer(128, x, encoder4)
    x = DecoderLayer(64, x, encoder3)
    x = DecoderLayer(64, x, encoder2)
    x = DecoderLayer(32,  x, encoder1)
    
    #Output
    if (n_classes == 1):
        x = Conv2D(n_classes, (1, 1), activation='sigmoid', padding='same')(x)
    else:
        x = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(x)
        
    model_output = x
    
    model = Model(model_input, model_output)
    return model


#Compile for binary data (pretraining)
def CompileBinary(model):
    loss = keras.losses.binary_crossentropy
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])


# Compile with the optimizer and the loss function
def Compile(model, loss='categorical_crossentropy', weight_loss=None):
    if (loss == 'weighted_categorical'):
        loss_final = weighted_categorical_crossentropy(weight_loss)
        model.compile(optimizer='adam',
                  loss=loss_final,
                  metrics=['accuracy', mean_dice])
        
    elif (loss == 'dice'):
        loss_final = dice_loss
        model.compile(optimizer='adam',
                  loss=loss_final,
                  metrics=['accuracy', mean_dice])
        
    elif (loss == 'categorical_crossentropy'):
        loss_final = keras.losses.categorical_crossentropy
        model.compile(optimizer='adam',
                  loss=loss_final,
                  metrics=['accuracy', mean_dice])
    return model

#Train a model with the image data generator
def Train(model, train_gen, val_gen, steps_per_epoch=100, batch_size=1, epochs=12):
    hist = model.fit(train_gen,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=16)
    
    results = [hist.history['val_accuracy'][-1], hist.history['val_mean_dice'][-1]]
    
    return hist, results

#Entrena distintos modelos usando cross validation y devuelve la accuracy media
def CrossValidation(model, train_data, train_labels, TrainArgs, ValArgs, 
                    loss='categorical_crossentropy', weight_loss=None,
                    steps_per_epoch=100, n_splits=3, epochs=12, batch_size=1):
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    Grupo = 0
    
    # Generate and compilate copies of the model
    models = [keras.models.clone_model(model) for i in range(n_splits)]
    for m in models:
        Compile(m, loss=loss, weight_loss=weight_loss)
        
    accuracies = []
    dices = []
    historials = []
    
    for train_indices, val_indices in kfold.split(train_data):
        print ('#########################################') 
        print (f'Cross Validation {Grupo + 1}/{n_splits}') 
        
        train_x_data = train_data[train_indices]
        train_y_data = train_labels[train_indices]
        
        train_data_fold = GenerateData(train_x_data, train_y_data, TrainArgs)
        
        val_x_data = train_data[val_indices]
        val_y_data = train_labels[val_indices]
        
        val_data_fold = GenerateData(val_x_data, val_y_data, ValArgs)
        
        historial, results = Train(models[Grupo], train_data_fold, val_data_fold, 
                          steps_per_epoch=steps_per_epoch, batch_size=batch_size, epochs=epochs)
        
        historials.append(historial)
        accuracies.append(results[0])
        dices.append(results[1])
        
        Grupo = Grupo +1
        
    best_network = dices.index(max(dices))
    mean_dice = sum(dices) / len(dices)
    mean_accuracy = sum(accuracies) / len(accuracies)
    
    print(f'Results: Accuracies: {accuracies} Dice: {dices}')
    print(f'Mean Dice: {mean_dice}')
    print(f'Mean Accuracy: {mean_accuracy}')
    
    results = [mean_accuracy, mean_dice]
    
    return historials[best_network], results

def Test(model, X_test, Y_test):
    predicciones = model.predict(X_test, batch_size=4)
    labels = np.argmax(Y_test, axis = -1)
    preds = np.argmax(predicciones, axis = -1)
    
    print(f"Y_test: {Y_test.shape} labels: {labels.shape} predicciones: mask: {predicciones.shape} {preds.shape}")
    
    accuracy = sum(labels.reshape((-1,)) == preds.reshape((-1,)))/len(labels.reshape((-1,)))
    dice = mean_dice(Y_test, predicciones)
    
    print(f"Accuracy={accuracy} Dice={dice}")
    
    for i in range(min(len(X_test), 5)):        
        ShowImage(predicciones[i,:,:,0], "Background")
        ShowImage(predicciones[i,:,:,1], "Blood cells")
        ShowImage(predicciones[i,:,:,2], "Bacteries")
        visualize(X_test[i], Y_test[i])
        visualize(X_test[i], predicciones[i])
        localaccuracy = sum((labels[i].reshape((-1,)) == preds[i].reshape((-1,))))/len(labels[i].reshape((-1,)))
        print(f"{i}: Accuracy={localaccuracy}")
        
    return accuracy

#Change the output layer of a model
def AdjustModel(model, n_classes):
    model_input = model.input
    
    #We eliminate the last layer
    x = model.layers[-2].output
    
    if (n_classes == 1):
        model_output = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)
    else:
        model_output = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(x)
    
    model = Model(model_input, model_output)
    
    return model
    
#Pretrain a model with the pre-train dataset
def PreTrain(model, pathtosave, name=""):
    X_train, Y_train, X_test, Y_test = LoadPretrainingData()
    
    model = AdjustModel(model, 1)
    CompileBinary(model)
    
    hist = model.fit(X_train, Y_train,
                        batch_size=4,
                        epochs=30,
                        verbose=1,
                        validation_data=(X_test, Y_test))
    
    name = name + " pre-train"
    ShowEvolution(name, hist)
    
    model.save(pathtosave)
    
    return model

#Load a model from memory. If n_classes is provided, the output layer is changed accordingly
def LoadModel(pathtosave, n_classes=None):
    model = keras.models.load_model(pathtosave)
    if (n_classes != None):
        model = AdjustModel(model, n_classes)
    return model


def main():
    
    X_train, Y_train, X_test, Y_test = LoadData()
    train_gen, val_gen, test_gen, TrainArgs, TestArgs = GetGenerators(
        X_train, Y_train, X_test, Y_test, data_augmentation=True, batch_size=4)
    
    
    experimentalResults = []
    
    def Experiment(name, model, useCrossValidation=True, TrainArgs_=None, TestArgs_=None, 
                   loss='categorical_crossentropy', weight_loss=None, epochs=5, steps_per_epoch=400, add_results=True):
        if (TrainArgs_ is None):
            TrainArgs_=TrainArgs
        if (TestArgs_ is None):
            TestArgs_=TestArgs
        
        if (useCrossValidation):
            hist, results = CrossValidation(model, X_train, Y_train, TrainArgs_, TestArgs_,
                                        loss=loss, weight_loss=weight_loss, steps_per_epoch=steps_per_epoch, epochs=epochs)
        else:
            hist, results = Train(model, train_gen, val_gen, steps_per_epoch=steps_per_epoch, batch_size=1, epochs=epochs)
        
        ShowEvolution(name, hist)
        
        #Add results 
        if (add_results):
            accuracy_four_decimals = format(results[0], '.4f')
            dice_four_decimals = format(results[1], '.4f')
            experimentalResults.append((f'{name}: {dice_four_decimals}%', results[1]))
        print(f"{name}: Accuracy = {accuracy_four_decimals} Dice = {dice_four_decimals}")
    
    ############################# DATA AUGMENTATION ##############################################
    def DataAugmentationTests():
        print("Test of data augmentation")
        experiment_steps_per_epoch = 50
        experiment_epochs = 5
        
        model = UNetClassic()
        Compile(model, loss='categorical_crossentropy')
        _,_,_, NoDATrainArgs, NoDATestArgs = GetGenerators(X_train, Y_train, X_test, Y_test, data_augmentation=False, batch_size=4)
        Experiment(f"No Data Augmentation", model, TrainArgs_=NoDATrainArgs, steps_per_epoch=experiment_steps_per_epoch, epochs = experiment_epochs)
        
        model = UNetClassic()
        Compile(model, loss='categorical_crossentropy')
        Experiment(f"Data Augmentation", model, steps_per_epoch=experiment_steps_per_epoch, epochs = experiment_epochs)
        
        PlotBars(experimentalResults, "Data Augmentation", "Dice")
        
    ############################# PRE-TRAIN ##############################################
    def PreTrainingTests():
        print("Test of pretraining")
        experiment_steps_per_epoch = 100
        experiment_epochs = 5
        
        model = LoadModel(pretrainedUNet, 3)
        Compile(model, loss='categorical_crossentropy')
        Experiment(f"Pre-Trained", model, steps_per_epoch=experiment_steps_per_epoch, epochs = experiment_epochs)
        
        model = UNetClassic()
        Compile(model, loss='categorical_crossentropy')
        Experiment(f"Not Pre-Trained", model, steps_per_epoch=experiment_steps_per_epoch, epochs = experiment_epochs)
        
        PlotBars(experimentalResults, "Pre-trained", "Dice")
        
    ############################# LOSS FUNCTIONS ##############################################
    def lossFunctionsTests():
        print("Test of loss functions")
        weights = calculateClassWeights(Y_train)
        print(weights)
        # Experiment with the shifts
        losses = ['weighted_categorical', 'dice', 'categorical_crossentropy']
        for l in losses:
          model = UNetClassic()
          Compile(model, loss=l, weight_loss = weights)
          _,_,_, TrainArgs, TestArgs = GetGenerators(X_train, Y_train, X_test, Y_test, batch_size=4)
          Experiment(f"Loss {l}", model, TrainArgs_=TrainArgs, steps_per_epoch=100, epochs = 5, loss =  l, weight_loss=weights)
        print(experimentalResults)
        PlotBars(experimentalResults, "Loss funcion", "Dice")
    
    ############################# UNET CLASSIC ##############################################
    def UnetClassicTest():
        print("Complete test of U-Net")
        #If the model is already saved
        # unet = LoadModel(pretrainedUNet, 3)
        
        unet = UNetClassic()
        unet = PreTrain(unet, pretrainedUNet, name="UNet Classic")
        unet = AdjustModel(unet, 3)
        
        Compile(unet, loss='categorical_crossentropy')
        print(unet.summary())
        Experiment("Unet", unet, useCrossValidation=False, steps_per_epoch=100, epochs=30)
        
        unet.save(savedUNet)
        Test(unet, X_test, Y_test)
    
    ############################# UNET CLASSIC ##############################################
    def Unetv2Test():
        print("Complete test of U-Netv2")
        #If the model is already saved
        # unetv2 = LoadModel(pretrainedUNetv2, 3)
        
        unetv2 = UNetV2()
        unetv2 = PreTrain(unetv2, pretrainedUNetv2, name="UNet v2")
        unetv2 = AdjustModel(unetv2, 3)
        
        Compile(unetv2, loss='categorical_crossentropy')
        print(unetv2.summary())
        # PreTrain(unetv2, pretrainedUNetv2, name="UNet v2")
        Experiment("Unet v2", unetv2, useCrossValidation=False, steps_per_epoch=100, epochs=30)
        
        unetv2.save(savedUNetv2)
        Test(unetv2, X_test, Y_test)
        
    # #Print some images
    # for i in range(20):
    #     ShowImage(X_train[0])
        
    # #Extract Percentages of the classes
    # ClassPercentage(Y_train) 
    
    #Pretrain models
    
    # unet = UNetClassic()
    # PreTrain(unet, pretrainedUNet, name="UNet Classic")
    
    # unetv2 = UNetV2()
    # PreTrain(unetv2, pretrainedUNetv2, name="UNet v2")
    
    #Experiments
    # DataAugmentationTests()
    # PreTrainingTests()
    # lossFunctionsTests()
    
    #Complete tests
    UnetClassicTest()
    Unetv2Test()

if __name__ == '__main__':
  main()