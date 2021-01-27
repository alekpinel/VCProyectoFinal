# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:46:12 2021

@author: alekp
"""

import numpy as np
import pandas as pd
from PIL import Image
import os
import cv2
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import random
from loss import *

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 16
NUM_CLASSES = 3
# IMG_PATH = './notebookdata/images/'
# MASK_PATH = './notebookdata/masks/'

# DATA_PATH = '../data/'

# LABEL_PATH = f'{DATA_PATH}masks/'
# ORIGINAL_IMAGES_PATH = f'{DATA_PATH}images/'

# IMG_SUB_PATH = f'{IMG_PATH}images/'
# MASK_SUB_PATH = f'{MASK_PATH}masks/'


# def ProcessData():
#     mask_files = os.listdir(MASK_PATH)
#     for mf in tqdm (mask_files):
#         mask_img = cv2.imread(os.path.join(MASK_PATH, mf), cv2.IMREAD_GRAYSCALE)
#         mask_img = np.around(tf.keras.utils.to_categorical(mask_img, NUM_CLASSES))
#         cv2.imwrite(os.path.join(MASK_SUB_PATH, mf), mask_img)
        
#         image_img = cv2.imread(os.path.join(IMG_PATH, mf), cv2.IMREAD_GRAYSCALE)
#         cv2.imwrite(os.path.join(IMG_SUB_PATH, mf), image_img)

datapath = "../data/" #Local

#Function that load and image and convert it to RGB if needed
def LoadImage(filename, color = True):
    if (color):
      img = cv2.imread(filename,cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

    return img

def ToCategoricalMatrix(data):
    originalShape = data.shape
    totalFeatures = data.max() + 1
    
    categorical = data.reshape((-1,))
    categorical = tf.keras.utils.to_categorical(categorical)
    data = categorical.reshape(originalShape + (totalFeatures,))
    return data

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

# ProcessData()

# os.system(f'mkdir {IMG_PATH}')
# os.system(f'cp -r {ORIGINAL_IMAGES_PATH} {IMG_PATH}')
# os.system(f'mkdir {MASK_PATH}')
# os.system(f'cp -r {LABEL_PATH} {MASK_PATH}')


    
def show_img(img, title=''):
    # given a numpy array, plot it
    vis = plt.imshow(img)
    plt.title(title)
    plt.show()
    
def show_imgs(imgs, titles=None):
    # show two images side by side, useful for segmentation projects
    fig = plt.figure(figsize=(15, 15))

    for i in range(len(imgs)):
        plt.subplot(1, len(imgs), i+1)
        if titles:
            plt.title(titles[i])
        plt.imshow(imgs[i])
    
    plt.show()

def label_to_image(m):
    # given a mask, turn it into an image, use for binary segmentations
    return tf.keras.preprocessing.image.array_to_img(m.reshape((m.shape[0], m.shape[1], 1)))

def output_to_image(o):
    # given model output o that is one hot encoded, turn it into an image, use for multi class segmentations
    mask_im = tf.argmax(o, axis=-1)
    mask_im = tf.keras.preprocessing.image.array_to_img(mask_im[..., tf.newaxis])
    
    return mask_im

def visualise_source(title): 
    # given an image title, visualise the source data
    test_img = cv2.imread(os.path.join(datapath + "images/", title + '.png'))
    mask_img = cv2.imread(os.path.join(datapath + "masks/", title + '.png'), cv2.IMREAD_GRAYSCALE)
    show_imgs([test_img, mask_img])
    
visualise_source('003')

# # detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# https://github.com/keras-team/keras/issues/3059#issuecomment-364787723
training_generation_args = dict(
#     width_shift_range=0.3,
#     height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    validation_split=0.1
)
train_image_datagen = ImageDataGenerator(**training_generation_args)
train_label_datagen = ImageDataGenerator(**training_generation_args)

X_train, Y_train, X_test, Y_test = LoadData()

# data load
training_image_generator = train_image_datagen.flow(
    X_train,
    # target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    # class_mode=None,
    subset='training',
    batch_size=BATCH_SIZE,
    seed=1
)
training_label_generator = train_label_datagen.flow(
    Y_train,
    # target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    # class_mode=None,
    subset='training',
    batch_size=BATCH_SIZE,
    # color_mode='grayscale',
    seed=1
)


# validation data load
validation_image_generator = train_image_datagen.flow(
    X_train,
    # target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    # class_mode=None,
    subset='validation',
    batch_size=BATCH_SIZE,
    seed=1
)
validation_label_generator = train_label_datagen.flow(
    Y_train,
    # target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    # class_mode=None,
    subset='validation',
    batch_size=BATCH_SIZE,
    # color_mode='grayscale',
    seed=1
)

train_generator = zip(training_image_generator, training_label_generator)
validation_generator = zip(validation_image_generator, validation_label_generator)

test_imgs, labels = train_generator.__next__()
show_imgs([test_imgs[0], labels[0]])

loss_weights = {
    0: 0,
    1: 0,
    2:0
}

for mask_img in tqdm(Y_train):
    # mask_img = cv2.imread(os.path.join(MASK_SUB_PATH, mf))
    classes = tf.argmax(mask_img, axis=-1).numpy()
    class_counts = np.unique(classes, return_counts=True)
    
    for c in range(len(class_counts[0])):
        loss_weights[class_counts[0][c]] += class_counts[1][c]

print(loss_weights)

total = sum(loss_weights.values())
for cl, v in loss_weights.items():
    # do inverse
    loss_weights[cl] = total / v
    
print(loss_weights)

# sys.exit()

w = [[loss_weights[0], loss_weights[1], loss_weights[2]]] * IMAGE_WIDTH
h = [w] * IMAGE_HEIGHT
loss_mod = np.array(h)

inputs = tf.keras.layers.Input(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])
x = inputs
# downstack
x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
x = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPooling2D((2, 2))(conv4)

x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)

# upstack
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Concatenate()([conv4, x])
x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Concatenate()([conv3, x])
x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Concatenate()([conv2, x])
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Concatenate()([conv1, x])
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(3, (1, 1), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs=x)

model.compile(optimizer='adam',
               loss='categorical_crossentropy',
              # loss=weighted_categorical_crossentropy(loss_mod),
              metrics=['accuracy', mean_dice],
              # loss_weights=loss_mod
              )

tf.keras.utils.plot_model(model, show_shapes=True)

print(model.summary())

model_history = model.fit(train_generator,
                          epochs=5,
                          steps_per_epoch=100,
                          validation_data=validation_generator,
                          validation_steps=9)

test_imgs, labels = validation_generator.__next__()
predictions = model.predict(test_imgs, use_multiprocessing=False)

for i in range(min(len(predictions), 5)):
    show_imgs(
        [test_imgs[i], labels[i], predictions[i]],
        ['Source Image', 'True Mask', 'Prediction']
    )
    
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()

# Loss 

plt.plot(model_history.history['val_loss'])
plt.plot(model_history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()