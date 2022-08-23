import torchvision.models.segmentation
import torch
import PIL
import torchvision.transforms as tf
import numpy as np


def get_device():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device == torch.device('cuda'):
        print('CUDA-enabled GPU is used')
    else:
        print('Warning: CPU is used, training will be very slow. Please consider using a CUDA-enabled GPU.')
    return device


def load_model(model_path, device):
    Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    Net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 1 class
    Net = Net.to(device)
    Net.load_state_dict(torch.load(model_path)) # Load net from 5000.torch
    return Net


def get_prediction(imagePath, Net, device): # Get prediction of an image
    height = width = 250
    transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor()])
    Img = PIL.Image.open(imagePath)  # load test image
    Img = np.array(Img)[:, :, 0:3]
    height_orgin, widh_orgin, d = Img.shape  # Get image original size
    Img = transformImg(Img)  # Transform to pytorch
    Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
    with torch.no_grad():
        Prd = Net(Img)['out']  # Run net
    Prd = torch.sigmoid(Prd)
    Prd = tf.Resize((height_orgin, widh_orgin))(Prd[0])  # Resize to origninal size
    # visualize Prd
    Prd = torch.squeeze(Prd)  # reduce dimension to (width,height)
    return Prd
