import sys

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

classes = ['apple']
num_classes = len(classes)
IMAGE_SIZE = 224

# convert data by specifying file from terminal
image = Image.open(sys.argv[1])
image = image.convert('RGB')
image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
data = np.asarray(image)
X = []
X.append(data)
X = np.array(X)

# load model
model = load_model('vgg16_transfer.h5')

# estimated result of the first data (multiple scores will be returned)
result = model.predict([X])[0]
predicted = result.argmax()
percentage = int(result[predicted] * 100)

print(classes[predicted], percentage)
