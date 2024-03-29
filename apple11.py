import glob

import numpy as np
from PIL import Image
from sklearn import model_selection

classes = ['ringo01']
num_classes = len(classes)
IMAGE_SIZE = 224  # Specified size of VGG16 Default input size in VGG16

X = []  # image file
Y = []  # correct label

for index, classlabel in enumerate(classes):
    photo_dir = classlabel
    files = glob.glob(photo_dir + '/*.jpeg')
    for i, file in enumerate(files):
        image = Image.open(file)
        # standardize to 'RGB'
        image = image.convert('RGB')
        # to make image file all the same size
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

# from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)
np.save('./image_files.npy', xy)
