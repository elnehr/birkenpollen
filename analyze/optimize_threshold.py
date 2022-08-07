from shutil import copyfile
import os
from get_area import get_area
import pandas as pd
import matplotlib.pyplot as plt
from get_masks import get_masks
from area_test import get_dice_coef
import PIL
import numpy as np



def test_threshold():
    df = pd.DataFrame(columns=['threshold', 'dice_coef'])
    for threshold in range(550, 650, 2):
        get_masks(threshold/1000)
        df.loc[len(df)] = [threshold, get_dice_coef()]
    return df


df2 = test_threshold()

# plot the dice score vs the threshold
plt.plot(df2['threshold'], df2['dice_coef'], 'ro')
plt.xlabel('threshold')
plt.ylabel('dice_coef')
plt.show()




