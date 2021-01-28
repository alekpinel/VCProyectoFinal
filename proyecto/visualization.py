import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize(image, mask, apply=False, title=None):
    mask = MaskToCategorical(mask)
    
    if isinstance(image, str):
        image = cv2.imread(image)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    if apply:
        mask_applied = image.copy()
        mask_applied[mask == 1] = [255, 0, 0]
        mask_applied[mask == 2] = [0, 255, 0]
        out = image.copy()
        mask_applied = cv2.addWeighted(mask_applied, 0.5, out, 0.5, 0, out)
    
    rgb_mask = np.zeros(image.shape, dtype=np.uint8)
    rgb_mask[mask==1] = [255,0,0]
    rgb_mask[mask==2] = [0,255,0]
    
    
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(1,3,1)
    plt.xticks([])
    plt.yticks([])
    
    ax1.imshow(image)
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(rgb_mask)
    plt.xticks([])
    plt.yticks([])

    
    if apply:
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(mask_applied)
        plt.xticks([])
        plt.yticks([])
    
    if (title != None):
        plt.title(title)
    
    plt.show()


def ShowEvolution(name, hist):
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
    # plt.legend(['Training accuracy','Validation accuracy'])
    # plt.title(name)
    # plt.show()
    
    if ('mean_dice' in hist.history):
        dice = hist.history['mean_dice']
        val_dice = hist.history['val_mean_dice']
        plt.plot(dice)
        plt.plot(val_dice)
        plt.legend(['Training accuracy','Validation accuracy', 'Training dice','Validation dice'])
    else:
        plt.legend(['Training accuracy','Validation accuracy'])
    
    plt.title(name)
    plt.show()

def MaskToCategorical(mask):
    if (len(mask.shape) == 3 and mask.shape[2] == 3):
        mask = np.argmax(mask, axis=-1)
    return mask

def VisualizeMask(mask, title=None):
    mask = MaskToCategorical(mask)
    
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask==1] = [255,0,0]
    rgb_mask[mask==2] = [0,255,0]
    ShowImage(rgb_mask, title)
    
    
#Shows an Image using Matplotlib 
def ShowImage(img, title=None):
    plt.imshow(img, cmap='gray')
    if (title != None):
        plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.show()
    
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
    
    
#Plot a graphic with values in the form (x, y)
def PlotResults(data, title, y_label):
    strings = [i[0] for i in data]
    x = [i for i in range(len(data))]
    y = [i[1] for i in data]
    
     
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel(y_label)
    
    fig.autofmt_xdate()
    
    x_labels=strings
    plt.xticks(x, x_labels)
    
    plt.bar(x, y)
    plt.show()