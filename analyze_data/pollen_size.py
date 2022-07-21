import os

import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['image', 'pixel_area'])

MaskFolder = "masks/"
ListMasks=os.listdir(os.path.join(MaskFolder))
ListMasks = [x for x in ListMasks if not x.startswith('.')]

for mask in ListMasks:
    Filled =  PIL.Image.open(os.path.join(MaskFolder, mask)).convert("L") # 0 = grayscale
    Filled = np.array(Filled)
    AnnMap = np.zeros(Filled.shape[0:2],np.float32)  # Create empty annotation map
    if Filled is not None:  AnnMap[ Filled  == 255 ] = 1

    #get area of the mask
    area = AnnMap.sum()
    df.loc[len(df)] = [mask, area]

df.to_csv('mask_area.csv', index=False)

#histogram of the area of the masks
plt.hist(df['pixel_area'], bins=100)
plt.show()