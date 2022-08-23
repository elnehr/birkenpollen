import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import PIL
from birkenpollen.train_segmentation.load import get_device, load_model, get_prediction

ImageFolder= os.path.join(os.path.dirname(__file__), '..', 'data/images/')
ListImages=os.listdir(os.path.join(ImageFolder)) # Create list of images
ListImages = [x for x in ListImages if not x.startswith('.')]

modelPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'train_segmentation/5000.torch'))  # Path to trained model
height=width=250

device = get_device()
Net = load_model(modelPath, device) # Load net from 5000.torch
Net.eval() # Set to evaluation mode
print("Loaded model")

def get_masks(threshold):
    for image in ListImages:
        image = image.replace('Ã¼', 'ü')
        imagePath = os.path.join(ImageFolder, image)
        Prd = get_prediction(imagePath, Net, device)

        boolean_mask = (Prd>threshold)

        # save boolean_mask to png
        boolean_mask = boolean_mask.cpu().detach().numpy()
        boolean_mask = boolean_mask.astype(np.uint8)
        boolean_mask = boolean_mask * 255
        boolean_mask = boolean_mask.astype(np.uint8)
        target_folder = os.path.join(os.path.dirname(__file__), '..', 'data/masks/')
        cv2.imencode(".png", boolean_mask)[1].tofile(os.path.join(target_folder, image))

if __name__ == "__main__":
    get_masks(0.56)

