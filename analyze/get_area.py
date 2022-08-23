import os
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_masks import get_masks


MaskFolder = os.path.join(os.path.dirname(__file__), '..', 'data/masks/')


def set_scale(): # open a pop up window to input the scale from pixel to micrometer (in this case 1 pixel = 1 micrometer)
    scale = input('Enter the scale in micrometer per pixel: ')
    return scale


def get_area(maskfolder):
    ListMasks = os.listdir(os.path.join(maskfolder))
    ListMasks = [x for x in ListMasks if not x.startswith('.')]
    df = pd.DataFrame(columns=['image', 'pixel_area'])
    for mask in ListMasks:
        Filled =  PIL.Image.open(os.path.join(maskfolder, mask)).convert("L") # 0 = grayscale
        Filled = np.array(Filled)
        AnnMap = np.zeros(Filled.shape[0:2],np.float32)  # Create empty annotation map
        if Filled is not None:  AnnMap[ Filled  == 255 ] = 1

        #get area of the mask
        area = AnnMap.sum()
        df.loc[len(df)] = [mask, area]
    return df

if __name__ == "__main__":
    get_masks(0.56)
    scale= set_scale()
    df = get_area(MaskFolder)
    df["area"] = df["pixel_area"] * float(scale)**2
    target_folder = os.path.join(os.path.dirname(__file__), '..', 'data/pollen_area.csv')
    df.to_csv(target_folder, index=False)
    print("Area saved to {}".format(target_folder))

    #histogram of the area of the masks
    #plt.hist(df['pixel_area'], bins=100)
    #plt.show()