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