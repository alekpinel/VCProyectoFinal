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

from loss import *
from visualization import *

#The local GPU used to run out of memory, so we limited the memory usage:
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Esta función pinta dos gráficas, una con la evolución
# de la función de pérdida en el conjunto de train y
# en el de validación, y otra con la evolución de la
# accuracy en el conjunto de train y el de validación.
# Es necesario pasarle como parámetro el historial del
# entrenamiento del modelo (lo que devuelve la
# función fit()).
def mostrarEvolucion(name, hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.title(name)
    plt.show()
    
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.title(name)
    plt.show()

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
    
#Get the generators from raw data
def GetGenerators(X_train, Y_train, X_test, Y_test, validation_split=0.1, batch_size=128, data_augmentation=False, seed=None):
    basic_generator_args = dict(
        validation_split=validation_split
    )
    
    data_augmentation_generator_args = dict(
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=5,
        
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        validation_split=validation_split
    )
    
    if (seed is None):
        seed = 1
    
    if (data_augmentation):
        data_generator_args = data_augmentation_generator_args
    else:
        data_generator_args = basic_generator_args
    
    train_image_datagen = ImageDataGenerator(**data_augmentation_generator_args)
    train_masks_datagen = ImageDataGenerator(**data_augmentation_generator_args)
    test_image_datagen = ImageDataGenerator()
    test_masks_datagen = ImageDataGenerator()
    
    # Training
    training_image_generator = train_image_datagen.flow(
        X_train,
        subset='training', batch_size=batch_size, seed=seed)
    
    training_masks_generator = train_masks_datagen.flow(
        Y_train,
        subset='training', batch_size=batch_size, seed=seed)
    
    train_gen = zip(training_image_generator, training_masks_generator)
    
    # Validation
    validation_image_generator = train_image_datagen.flow(
        X_train,
        subset='validation', batch_size=batch_size, seed=seed)
    
    validation_label_generator = train_masks_datagen.flow(
        Y_train,
        subset='validation', batch_size=batch_size, seed=seed)
    
    val_gen = zip(validation_image_generator, validation_label_generator)
    
    # Test
    test_image_generator = test_image_datagen.flow(
        X_test,
        batch_size=batch_size, seed=seed)
    
    test_label_generator = test_masks_datagen.flow(
        Y_test,
        batch_size=batch_size, seed=seed)
    
    test_gen = zip(test_image_generator, test_label_generator)
    
    return train_gen, val_gen, test_gen
    


def ToCategoricalMatrix(data):
    originalShape = data.shape
    totalFeatures = data.max() + 1
    
    categorical = data.reshape((-1,))
    categorical = to_categorical(categorical)
    data = categorical.reshape(originalShape + (totalFeatures,))
    return data

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
    
    PlotBars(data, "Class Percentages", "Percent")

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

#Classic implementation of UNet
def UNetV2(input_shape=(256, 256, 3), n_classes=3):
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
        x = Concatenate()([x, skip])
        # x = BatchNormalization()(x)
        x = Conv2DTranspose(filters, (3, 3), activation='relu', padding='same')(x)
        x = Conv2DTranspose(filters, (3, 3), activation='relu', padding='same')(x)
        return x
    
    #Input
    x = Input(input_shape)
    model_input = x
    
    #Encoder
    x, encoder1 = EncoderLayer(32,  x)
    x, encoder2 = EncoderLayer(64, x)
    x, encoder3 = EncoderLayer(128, x)
    x, encoder4 = EncoderLayer(256, x)
    
    #Centre
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    #Decoder
    x = DecoderLayer(256, x, encoder4)
    x = DecoderLayer(128, x, encoder3)
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
def Compile(model, loss='weighted_categorical', weight_loss=None):
    if (loss == 'weighted_categorical'):
        loss_final = weighted_categorical_crossentropy(weight_loss)
        model.compile(optimizer='adam',
                  loss=loss_final,
                  metrics=['accuracy', mean_dice])
        
    elif (loss == 'dice'):
        loss_final = standard_dice
        model.compile(optimizer='adam',
                  loss=loss_final,
                  metrics=['accuracy', mean_dice])
        
    elif (loss == 'categorical_crossentropy'):
        loss_final = keras.losses.categorical_crossentropy
        model.compile(optimizer='adam',
                  loss=loss_final,
                  metrics=['accuracy', mean_dice])
    return model

def Train(model, train_gen, val_gen, steps_per_epoch=100, batch_size=1, epochs=12):
    
    hist = model.fit(train_gen,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=9)
    return hist

def Test(model, X_test, Y_test):
    predicciones = model.predict(X_test)
    labels = np.argmax(Y_test, axis = -1)
    preds = np.argmax(predicciones, axis = -1)
    
    print(f"Y_test: {Y_test.shape} labels: {labels.shape} predicciones: mask: {predicciones.shape} {preds.shape}")
    
    accuracy = sum(labels.reshape((-1,)) == preds.reshape((-1,)))/len(labels.reshape((-1,)))
    
    print(f"Accuracy={accuracy}")
    
    for i in range(len(X_test)):
        print(f"Maximos Background {np.max(predicciones[i,:,:,0].reshape((-1,)))}")
        print(f"Maximos Blood cells {np.max(predicciones[i,:,:,1].reshape((-1,)))}")
        print(f"Maximos Bacteries {np.max(predicciones[i,:,:,2].reshape((-1,)))}")
        
        ShowImage(predicciones[i,:,:,0], "Background")
        ShowImage(predicciones[i,:,:,1], "Blood cells")
        ShowImage(predicciones[i,:,:,2], "Bacteries")
        visualize(X_test[i], Y_test[i])
        visualize(X_test[i], predicciones[i])
        localaccuracy = sum((labels[i].reshape((-1,)) == preds[i].reshape((-1,))))/len(labels[i].reshape((-1,)))
        print(f"{i}: Accuracy={localaccuracy}")
        
    return accuracy

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

#Load a model from memory. If n_classes is provided, the output layer is changed accordingly
def LoadModel(pathtosave, n_classes=None):
    model = keras.models.load_model(pathtosave)
    if (n_classes != None):
        model = AdjustModel(model, n_classes)
    return model

def ToyModel(input_shape=(256, 256, 3), n_classes=3):
    #Input
    x = Input(input_shape)
    model_input = x
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(x)
    model_output = x
    model = Model(model_input, model_output)
    return model


def main():
    
    w_list = {0.0: 1.0, 1.0:7.0, 2.0:89.0}
    numbers = [i + 1 for i in range(10)]
    
    print(f"numbers: {numbers}")
    
    print(f"w_list: {w_list}")
    
    for n, w in zip(numbers, w_list):
        print(f"n:{n} w:{w_list[w]}")
    
    return 0
    
    X_train, Y_train, X_test, Y_test = LoadData()
    train_gen, val_gen, test_gen = GetGenerators(X_train, Y_train, X_test, Y_test,
                                                  data_augmentation=True,
                                                  batch_size=4)
    
    
    # test_imgs, labels = train_gen.__next__()
    # print(len(test_imgs))
    
    # return 0
    
    # for i in range(10):
    #     test_imgs, labels = train_gen.__next__()
    #     visualize(test_imgs[0], labels[0])
    
    # return 0
    
    # print(X_train.shape)
    # print(Y_train.shape)
    # print(X_test.shape)
    # print(Y_test.shape)
    
    # ClassPercentage(Y_train)
    
    #Pretrain block
    # unet = UNetV2()
    # PreTrain(unet, pretrainedUNetv2, name="UNet v2")
    # return 0
    
    class_weights = calculateClassWeights(Y_train)
    # class_weights = np.sqrt(class_weights)
    # class_weights = np.array([6.0, 14.0, 78.0])
    # class_weights = np.array([1.0, 1.0, 1.0])
    # class_weights = np.array([10.0, 12.0, 76.0])
    
    # print(weight_loss)
    
    # model = LoadModel(pretrainedUNetv2, 3)
    model = UNetV2()
    # model = UNetClassic()
    
    print(model.summary())
    
    # Compile(model, loss='categorical_crossentropy')
    Compile(model, loss='weighted_categorical', weight_loss=class_weights)
    
    # Test(model, X_train[:1], Y_train[:1])
    
    steps_per_epoch = 100
    
    hist = Train(model, train_gen, val_gen, steps_per_epoch=steps_per_epoch, batch_size=1, epochs=10)
    
    ShowEvolution("UNet", hist)
    
    # model.save(tempUNet)
    # model.save(savedUNet)
    
    print(f"Class Weights: {class_weights}")
    Test(model, X_train[:5], Y_train[:5])
    
    
    
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