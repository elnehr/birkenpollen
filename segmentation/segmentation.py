import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
import os
import numpy as np
modelPath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'train_segmentation/4500.torch'))  # Path to trained model
imagePath = "test2.png"  # Test image
height=width=250
transformImg = tf.Compose([tf.ToPILImage(), tf.Resize((height, width)), tf.ToTensor()])  # tf.Resize((300,600)),tf.RandomRotation(145)])#

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # Check if there is GPU if not set trainning to CPU (very slow)
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 1 classes
Net = Net.to(device)  # Set net to GPU or CPU
Net.load_state_dict(torch.load(modelPath)) # Load trained model
Net.eval() # Set to evaluation mode
Img = cv2.imread(imagePath) # load test image
height_orgin , widh_orgin ,d = Img.shape # Get image original size
plt.imshow(Img[:,:,::-1])  # Show image
plt.show()
Img = transformImg(Img)  # Transform to pytorch
Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
with torch.no_grad():
    Prd = Net(Img)['out']  # Run net
Prd = torch.sigmoid(Prd)
print(Prd)
print(Prd.shape)
Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0]) # Resize to origninal size
#visualize Prd
Prd = torch.squeeze(Prd) #reduce dimension to (width,height)

seg = Prd.cpu().detach().numpy()  # Get  prediction classes

plt.imshow(seg)  # display image
plt.colorbar()
plt.show()

#plt.imshow(np.where(seg<0.34,1,0))