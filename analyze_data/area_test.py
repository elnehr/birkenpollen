from shutil import copyfile
import os
from pollen_size import get_area
import pandas as pd
import matplotlib.pyplot as plt
from get_masks import get_masks
import PIL
import numpy as np

masks_model = "masks/"
masks_manual = "masks_manual/"


df_model = get_area(masks_model)
df_manual = get_area(masks_manual)

# merge the two dataframes, name pixel_area_model and pixel_area_manual
df = pd.merge(df_model, df_manual, on='image', how='outer')
df.rename(columns={'pixel_area_x': 'pixel_area_model', 'pixel_area_y': 'pixel_area_manual'}, inplace=True)


# compare the model and the manual masks
plt.scatter(df['pixel_area_model'], df['pixel_area_manual'], label='model vs manual')
plt.plot([0, df['pixel_area_model'].max()], [0, df['pixel_area_model'].max()], 'k--')
plt.xlabel('pixel_area_model')
plt.ylabel('pixel_area_manual')
plt.show()




def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001 # to avoid division by 0
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

MaskFolder = "masks/"
masks_manual_folder = "masks_manual/"
ListMasks=os.listdir(os.path.join(MaskFolder))
ListMasks = [x for x in ListMasks if not x.startswith('.')]

def get_dice_score():
    df = pd.DataFrame(columns=['image', 'dice_coef'])
    for mask in ListMasks:
        Filled =  PIL.Image.open(os.path.join(MaskFolder, mask)).convert("L") # 0 = grayscale
        Filled = np.array(Filled)
        AnnMap = np.zeros(Filled.shape[0:2],np.float32)  # Create empty annotation map
        if Filled is not None:  AnnMap[ Filled  == 255 ] = 1

        Filled_manual =  PIL.Image.open(os.path.join(masks_manual_folder, mask)).convert("L") # 0 = grayscale
        Filled_manual = np.array(Filled_manual)
        AnnMap_manual = np.zeros(Filled_manual.shape[0:2],np.float32)  # Create empty annotation map
        if Filled_manual is not None:  AnnMap_manual[ Filled_manual  == 255 ] = 1

        #calculate the dice score
        dice_score = dice_coef(AnnMap, AnnMap_manual)
        df.loc[len(df)] = [mask, dice_score]
        return df['dice_coef'].mean()


def test_threshold():
    df = pd.DataFrame(columns=['threshold', 'dice_coef'])
    for threshold in range(550, 650, 2):
        get_masks(threshold/1000)
        df.loc[len(df)] = [threshold, get_dice_score()]
    return df


df2 = test_threshold()

# plot the dice score vs the threshold
plt.plot(df2['threshold'], df2['dice_coef'], 'ro')
plt.xlabel('threshold')
plt.ylabel('dice_coef')
plt.show()




