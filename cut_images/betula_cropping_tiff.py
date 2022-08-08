import json
import os
from PIL import Image, ImageSequence
import pandas as pd


# this finds our json files
path_to_json = 'coordinates/'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# here I define my pandas Dataframe with the columns I want to get from the json
jsons_data = pd.DataFrame(columns=['imagePath', 'objectnr', 'label', 'points'])

# we need both the json and an index number so use enumerate()
for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)

    for i, shape in enumerate(json_text['shapes']):
        print(index, i)
        imagePath = json_text['imagePath']
        objectnr = i
        label = json_text['shapes'][i]['label']
        points = json_text['shapes'][i]['points']
        jsons_data.loc[len(jsons_data)] = [imagePath, objectnr, label, points]
        print(jsons_data)

df = jsons_data

df['imagePath'] = df['imagePath'].str.replace('Ã¼', 'ü')

for index, row in df.iterrows():
    x1 = row[3][0][0]
    y1 = row[3][0][1]
    x2 = row[3][1][0]
    y2 = row[3][1][1]
    img = Image.open("betula_jpgs/" + row['imagePath'])

    for i, page in enumerate(ImageSequence.Iterator(img)):
        img = page.crop((x1, y1, x2, y2))

        path = 'cropped_imgs/' + row['imagePath'].split('.')[0] + "/"
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)

        img.save(path + row['imagePath'].split('.')[0] + "_n" + str(row[1]) + "_p" + str(i) + ".png")




