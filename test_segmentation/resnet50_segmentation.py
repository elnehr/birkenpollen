from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as F
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms as T

import torch
import numpy as np
import matplotlib.pyplot as plt


## Beispiel Erkennung von Hunden

img = read_image("hund.jpg")

# Step 1: Initialize model with the best available weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights)
model.eval()


# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)  # batch size 1

# Step 4: Use the model and visualize the prediction
prediction = model(batch)["out"]
normalized_masks = prediction.softmax(dim=1)
class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

class_dim = 1
boolean_dog_masks = (normalized_masks.argmax(class_dim) == class_to_idx['dog'])
print(f"shape = {boolean_dog_masks.shape}, dtype = {boolean_dog_masks.dtype}")
F.to_pil_image(boolean_dog_masks.float()).show()


# image with mask overlay
transform  = T.Resize((520,781))
img = transform(img)

dogs_with_masks =  draw_segmentation_masks(img, masks=boolean_dog_masks, alpha=0.7)

#show(dogs_with_masks)
F.to_pil_image(dogs_with_masks).show()


