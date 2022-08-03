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
import torchvision.transforms as T
import PIL

ImageFolder= os.path.join(os.path.dirname( __file__ ), '..', 'data/images/')
ListImages = os.listdir(os.path.join(ImageFolder)) # Create list of images
ListImages = [x for x in ListImages if not x.startswith('.')]

modelPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'train_segmentation/5000.torch'))  # Path to trained model
height=width=250
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor()])  # tf.Resize((300,600)),tf.RandomRotation(145)])#

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 1 classes
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath)) # Load trained model
Net.eval() # Set to evaluation mode

def image_with_mask(imagePath):
    Img = PIL.Image.open(imagePath) # load test image
    Img = np.array(Img)[:, :, 0:3]
    height_orgin , widh_orgin ,d = Img.shape # Get image original size
    plt.imshow(Img[:,:,::-1])  # Show image
    plt.show()
    Img = transformImg(Img)  # Transform to pytorch
    Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
    with torch.no_grad():
        Prd = Net(Img)['out']  # Run net
    Prd = torch.sigmoid(Prd)
    Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0]) # Resize to origninal size
    #visualize Prd
    Prd = torch.squeeze(Prd) #reduce dimension to (width,height)

    seg = Prd.cpu().detach().numpy()  # Get  prediction classes

    plt.imshow(seg)  # display image
    #plt.imshow(np.where(seg>0.6,1,0))  # display image
    plt.colorbar()
    plt.show()

    boolean_mask = (Prd>0.44)
    img = read_image(imagePath)
    transform  = T.Resize((height_orgin , widh_orgin))
    img = transform(img)
    pollen_with_mask = draw_segmentation_masks(img, masks=boolean_mask, alpha=0.3)
    # return as png
    return F.to_pil_image(pollen_with_mask)

for image in ListImages:
    image = image.replace('Ã¼', 'ü')
    img = image_with_mask(os.path.join(ImageFolder, image))
    target_folder = os.path.join(os.path.dirname( __file__ ), '..', 'data/images_with_masks/')
    img.save(target_folder + image)