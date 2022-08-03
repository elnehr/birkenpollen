# Semantic Segmentatation to Measure Pollen Area
This is a tool to measure the area of pollen in images. It uses a semantic segmentation model and is partly 
based on the following [tutorial](https://expoundai.wordpress.com/2019/08/30/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch/).


## Get started


## Picture Format

The pictures used should contain a single pollen in the center of the image that isn't touching the image borders. Parts of other Pollen and other pollution in the picture does not 
pose a problem.  The file format must be a .png file.

Examples for usable pictures:
![My Image](sample_imgs/ok1.png)
![My Image](sample_imgs/ok2.png)
![My Image](sample_imgs/ok3.png)


Examples for not usable pictures:
![My Image](sample_imgs/bad1.png)
![My Image](sample_imgs/bad2.png)
![My Image](sample_imgs/bad3.png)
![My Image](sample_imgs/bad4.png)

## Measure Pollen Area

1. Put the images of the pollen that should be analyzed in data/images/

2. Run analyze/get_area.py

3. When the message 'Enter the scale in micrometer per pixel:' appears, type in the scale for your pictures and press enter. E.g. if one pixel in the image correspond to 0.5 μm you would enter 0.5

4. The data is saved as data/pollen_area.csv with the format: image, pixel_area, area. The masks created in the process are saved in data/masks. 

## Measure Model Performance on your dataset

1. Follow the steps to measure pollen area.

2. Put the manually segmented masks in the data/masks_manual/ folder. The masks should have a tranparent background and white when a pixel belongs to the pollen. E.g. [GIMP](https://www.gimp.org/) can be used to create the masks.

3. Run analyze/area_test.py

4. A scatter plot to compare the area from the masks is compared to the area of the manually created masks. Further, 
area of the masks manually and by the model created and the [dice coefficent](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2) are saved to data/model_test.csv

## Fine-tune the Model
1. Put the training images in data/images/

2. Put the manually segmented masks in the data/masks_manual/ folder. The masks should have a tranparent background and white when a pixel belongs to the pollen. E.g. [GIMP](https://www.gimp.org/) can be used to create the masks.

3. Run train_segmentation/train.py

## Acknowledgments

