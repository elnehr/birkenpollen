import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import PIL

Learning_Rate=1e-5
width=height=250 # image width and height
batchSize=3

TrainFolder="Imgs/"
ListImages=os.listdir(os.path.join(TrainFolder)) # Create list of images
#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor()])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(): # First lets load random image and  the corresponding annotation
    idx=np.random.randint(0,len(ListImages)) # Select random image
    Img=PIL.Image.open(os.path.join(TrainFolder, ListImages[idx]))
    Img= np.array(Img)[:,:,0:3]
    Filled =  PIL.Image.open(os.path.join("Masken", ListImages[idx])).convert("L") # 0 = grayscale
    Filled = np.array(Filled)
    AnnMap = np.zeros(Img.shape[0:2],np.float32)  # Create empty annotation map
    if Filled is not None:  AnnMap[ Filled  == 1 ] = 2  # Fill annotation map with 1 for filled pixels
    Img=transformImg(Img)
    AnnMap=transformAnn(AnnMap)
    return Img,AnnMap
#--------------Load batch of images-----------------------------------------------------
def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,3,height,width])
    ann = torch.zeros([batchSize, 1, height, width])
    for i in range(batchSize):
        images[i],ann[i]=ReadRandomImage()
    return images, ann
#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
#----------------Train--------------------------------------------------------------------------
def train():
    for itr in range(1000): # Training loop
       images,ann=LoadBatch() # Load taining batch
       images=torch.autograd.Variable(images,requires_grad=False).to(device) # Load image
       ann = torch.autograd.Variable(ann, requires_grad=False).to(device) # Load annotation
       Pred=Net(images)['out'] # make prediction
       Pred=torch.sigmoid(Pred)
       Net.zero_grad()
       criterion = torch.nn.BCELoss() # Set loss function
       Loss=criterion(Pred,ann.float()) # Calculate cross entropy loss
       Loss.backward() # Backpropogate loss
       optimizer.step() # Apply gradient descent change to weight
       seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()  # Get  prediction classes
       print(itr,") Loss=",Loss.data.cpu().numpy())
       if itr % 100 == 0: #Save model weight once every 100 steps permenant file
            print("Saving Model" +str(itr) + ".torch")
            torch.save(Net.state_dict(),   str(itr) + ".torch")

train()

print(os.path.join(TrainFolder, ListImages[4]))
print(PIL.Image.open(os.path.join(TrainFolder, ListImages[4])))
