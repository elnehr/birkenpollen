import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import PIL

ImageFolder="images/"
ListImages=os.listdir(os.path.join(ImageFolder)) # Create list of images
ListImages = [x for x in ListImages if not x.startswith('.')]

modelPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'train_segmentation/500.torch'))  # Path to trained model
height=width=250
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor()])  # tf.Resize((300,600)),tf.RandomRotation(145)])#

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 1 classes
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath)) # Load trained model
Net.eval() # Set to evaluation mode
print("Loaded model")

for image in ListImages:
    print(image)
    image = image.replace('Ã¼', 'ü')
    print(ImageFolder + image)
    Img = PIL.Image.open(os.path.join(ImageFolder, image)) # load test image
    Img = np.array(Img)[:, :, 0:3] # remove alpha channel
    height_orgin , widh_orgin ,d = Img.shape # Get image original size
    Img = transformImg(Img)  # Transform to pytorch
    Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
    with torch.no_grad():
        Prd = Net(Img)['out']  # Run net
    Prd = torch.sigmoid(Prd)
    Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0]) # Resize to original size
    #visualize Prd
    Prd = torch.squeeze(Prd) #reduce dimension to (width,height)

    boolean_mask = (Prd>0.45)

    # save boolean_mask to png
    boolean_mask = boolean_mask.cpu().detach().numpy()
    boolean_mask = boolean_mask.astype(np.uint8)
    boolean_mask = boolean_mask * 255
    boolean_mask = boolean_mask.astype(np.uint8)
    cv2.imencode(".png", boolean_mask)[1].tofile(os.path.join("masks/", image))



