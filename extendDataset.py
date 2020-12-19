from PIL import Image
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
def extendDataset(image_folder_path, extend_by):
    parent = image_folder_path
    X = []
    y = []
    i = 0

    image_gen = ImageDataGenerator(rotation_range=5,
                                   rescale = False,
                                   shear_range = 0.2,
                                   fill_mode='reflect',
                                   horizontal_flip=False,
                                   vertical_flip=False,
                                   brightness_range=[0.5, 1.5])
    
    for loc in os.listdir(parent):
        i += 1
        print(loc)
        im = Image.open(image_folder_path+loc)   
        im = im.resize((224, 224))
        im_array = np.asarray(im)
        X.append(im_array)
        ID = loc.split("_")
        y.append(ID[0]+"_"+ID[1])
        iter = image_gen.flow(np.expand_dims(im, 0))

        for _ in range(extend_by):
            X.append(np.asarray(next(iter)[0].astype(np.uint8)))
            # ID = loc.split("_")
            y.append(ID[0]+"_"+ID[1])

    X = np.asarray(X)
    y = np.asarray(y)
    return [X, y]