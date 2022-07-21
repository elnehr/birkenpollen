import os
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import PIL
import pandas as pd
import matplotlib.pyplot as plt

Learning_Rate=1e-5
width=height=250 # image width and height
batchSize=3

TrainFolder="Imgs/"
ListImages=os.listdir(os.path.join(TrainFolder)) # Create list of images
#split the list randomly into 80% train and 20% test images
train_list = ListImages[0:int(len(ListImages)*0.8)]
test_list = ListImages[int(len(ListImages)*0.8):]

#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor()])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(image_list): # First lets load random image and  the corresponding annotation
    idx=np.random.randint(0,len(image_list)) # Select random image
    Img=PIL.Image.open(os.path.join(TrainFolder, image_list[idx]))
    Img= np.array(Img)[:,:,0:3]
    Filled =  PIL.Image.open(os.path.join("Masken", image_list[idx])).convert("L") # 0 = grayscale
    Filled = np.array(Filled)
    AnnMap = np.zeros(Img.shape[0:2],np.float32)  # Create empty annotation map
    if Filled is not None:  AnnMap[ Filled  == 255 ] = 1  # Fill annotation map with 1 for filled pixels
    Img=transformImg(Img)
    AnnMap=transformAnn(AnnMap)
    return Img,AnnMap


#--------------Load batch of images-----------------------------------------------------
def LoadBatch(image_list): # Load batch of images
    images = torch.zeros([batchSize,3,height,width])
    ann = torch.zeros([batchSize, 1, height, width])
    for i in range(batchSize):
        images[i],ann[i]=ReadRandomImage(image_list)
    return images, ann
#--------------Load and set net and optimizer-------------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 2 classes
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
#----------------Train--------------------------------------------------------------------------

#create pandas dataframe to save loss values
df = pd.DataFrame(columns=['Iteration', 'Train Loss', 'Test Loss'])

def test(): # compute test loss
    images,ann=LoadBatch(test_list)
    images=torch.autograd.Variable(images,requires_grad=False).to(device)
    ann = torch.autograd.Variable(ann, requires_grad=False).to(device)
    Pred=Net(images)['out']
    Pred=torch.sigmoid(Pred)
    criterion = torch.nn.BCELoss()
    Loss=criterion(Pred,ann.float())
    test_loss = Loss.data.cpu().numpy()
    return test_loss

def train():
    for itr in range(501): # Training loop
       images,ann=LoadBatch(train_list) # Load taining batch
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
       train_loss = Loss.data.cpu().numpy() # Get loss
       print(itr,") Loss=",Loss.data.cpu().numpy())
       if itr % 10 == 0: # Save loss values every 10 iterations
            test_loss = test()
            df.loc[len(df)] = [itr, train_loss, test_loss]

       if itr % 500 == 0: #Save model weight once every 500 steps permenant file
            print("Saving Model" +str(itr) + ".torch")
            torch.save(Net.state_dict(),   str(itr) + ".torch")

train()

# visualize loss values
plt.plot(df['Iteration'], df['Train Loss'], label='Train Loss')
plt.plot(df['Iteration'], df['Test Loss'], label='Test Loss')
plt.legend()
plt.show()



