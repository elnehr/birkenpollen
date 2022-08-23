import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
from torchvision.io.image import read_image
from birkenpollen.train_segmentation.load import get_device, load_model, get_prediction
import torchvision.transforms as T
import PIL


ImageFolder= os.path.join(os.path.dirname( __file__ ), '..', 'data/images/')
ListImages = os.listdir(os.path.join(ImageFolder)) # Create list of images
ListImages = [x for x in ListImages if not x.startswith('.')]

modelPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'train_segmentation/5000.torch'))  # Path to trained model
height=width=250

device = get_device()
Net = load_model(modelPath, device) # Load net from 5000.torch
Net.eval() # Set to evaluation mode

def image_with_mask(imagePath):
    Prd = get_prediction(imagePath, Net, device)

    #plt.imshow(seg)  # display image
    #plt.imshow(np.where(seg>0.6,1,0))  # display image
    #plt.colorbar()
    #plt.show()

    boolean_mask = (Prd>0.56)
    img = read_image(imagePath)
    pollen_with_mask = draw_segmentation_masks(img, masks=boolean_mask, alpha=0.3)
    return F.to_pil_image(pollen_with_mask)


for image in ListImages:
    image = image.replace('Ã¼', 'ü')
    print(image)
    img = image_with_mask(os.path.join(ImageFolder, image))
    target_folder = os.path.join(os.path.dirname( __file__ ), '..', 'data/images_with_masks/')
    img.save(target_folder + image)