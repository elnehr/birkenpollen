import os
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import PIL
import pandas as pd
import matplotlib.pyplot as plt

Learning_Rate=1e-5
width=height=250 # image width and height
batchSize=3

TrainFolder = os.path.join(os.path.dirname(__file__), '..', 'data/images/')
MaskFolder = os.path.join(os.path.dirname(__file__), '..', 'data/masks/')
ListImages = os.listdir(os.path.join(TrainFolder)) # Create list of images
ListImages = [x for x in ListImages if not x.startswith('.')]

train_size = int(0.8 * len(ListImages))
test_size = len(ListImages) - train_size
train_list, test_list = torch.utils.data.random_split(ListImages, [train_size, test_size])

#----------------------------------------------Transform image-------------------------------------------------------------------
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor()])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])
#---------------------Read image ---------------------------------------------------------
def ReadRandomImage(image_list): # First lets load random image and  the corresponding annotation
    idx=np.random.randint(0,len(image_list))  # Select random image
    Img=PIL.Image.open(os.path.join(TrainFolder, image_list[idx]))
    Img= np.array(Img)[:,:,0:3]
    Filled =  PIL.Image.open(os.path.join(MaskFolder, image_list[idx])).convert("L")  # 0 = grayscale
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
if device == torch.device('cuda'):
    print('CUDA-enabled GPU is used')
else:
    print('Warning: CPU is used, training will be very slow. Please consider using a CUDA-enabled GPU.')
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
Net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 1 class
Net = Net.to(device)
Net.load_state_dict(torch.load('5000.torch')) # Load net from 5000.torch
optimizer = torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 5001)
#----------------Train--------------------------------------------------------------------------

#create pandas dataframe to save loss values
df = pd.DataFrame(columns=['Iteration', 'Train Loss', 'Test Loss', 'square_diff'])

def test():  # compute test loss
    images,ann=LoadBatch(test_list)
    images=torch.autograd.Variable(images,requires_grad=False).to(device)
    ann = torch.autograd.Variable(ann, requires_grad=False).to(device)
    Pred=Net(images)['out']
    Pred=torch.sigmoid(Pred)
    criterion = torch.nn.BCELoss()
    Loss=criterion(Pred,ann.float())
    test_loss = Loss.data.cpu().numpy()
    return test_loss

def visualize_loss():  # visualize loss values as a line plot
    plt.plot(df['Iteration'], df['Train Loss'], label='Train Loss')
    plt.plot(df['Iteration'], df['Test Loss'], label='Validation Loss')
    plt.legend()
    plt.show()
    plt.plot(df['Iteration'], df['square_diff'], label='Sum of squared differences')
    plt.legend()
    plt.show()

def area_test():  # compares area of predicted mask and actual mask in test set
    images,ann=LoadBatch(test_list)
    images=torch.autograd.Variable(images,requires_grad=False).to(device)
    ann = torch.autograd.Variable(ann, requires_grad=False).to(device)
    Pred = Net(images)['out']
    Pred = torch.sigmoid(Pred)
    Pred = Pred.data.cpu().numpy()
    ann = ann.data.cpu().numpy()
    area_diff = np.sum(np.square(Pred-ann))
    return area_diff


def train():  # train network
    for itr in range(5001): # Training loop with 5001 iterations
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
       scheduler.step() # Update learning rate
       train_loss = Loss.data.cpu().numpy() # Get loss
       print(itr,") Loss=",Loss.data.cpu().numpy())
       if itr % 10 == 0: # Save loss values every 10 iterations
            test_loss = test()
            square_diff = area_test()
            df.loc[len(df)] = [itr, train_loss, test_loss, square_diff]

       if itr % 100 == 0:
            visualize_loss()

       if itr % 500 == 0: #Save model weight once every 500 steps permenant file
            print("Saving Model" +str(itr) + ".torch")
            torch.save(Net.state_dict(),   str(itr) + ".torch")
            df.to_csv("loss.csv", index=False) # Save loss values to csv file



# save test_list to csv file
#test_list_csv = pd.DataFrame(list(test_list))
#test_list_csv.to_csv("test_list.csv", index=False)

train() # Train network