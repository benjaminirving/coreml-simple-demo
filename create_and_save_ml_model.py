"""
 - Load pretrained model
 - Get class labels for imagenet
 - Save ml model using coremltools

"""

from __future__ import print_function

import urllib
import os
import json

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt

import coremltools


def get_imagenet_class_labels():
    """
    Get the imagenet class index

    :return:
    """

    # Link to imagenet class labels provided by Keras
    class_label_path = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

    if not os.path.isfile("imagenet.json"):
        urllib.urlretrieve(class_label_path, "imagenet.json")
    with open("imagenet.json") as json_data:
        d = json.load(json_data)
        print(d)

    # Create a list of the class labels
    class_labels = []
    for ii in range(len(d.keys())):
        class_labels.append(d[str(ii)][1].encode('ascii', 'ignore'))

    return class_labels


# Load a model.
model = ResNet50(weights='imagenet')

# We could do some extra training etc here

# Classify an example
img_path = 'img/IMG_0.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict using the model
preds = model.predict(x)
pred_name = decode_predictions(preds, top=3)[0]

# Print predicted labels
print('Predicted:', pred_name)

# Show figure
x1 = imread(img_path)
plt.imshow(np.squeeze(x1))
plt.title('Predictions:\n {}: {}%,\n {}: {}%,\n {}:{}%'.format(
    pred_name[0][1], np.around(pred_name[0][2]*100),
    pred_name[1][1], np.around(pred_name[1][2]*100),
    pred_name[2][1], np.around(pred_name[2][2]*100)))
plt.axis('off')
plt.show()

# Convert to mlmodel
class_labels = get_imagenet_class_labels()
coreml_model = coremltools.converters.keras.convert(model,
                                                    input_names='data',
                                                    image_input_names='data',
                                                    class_labels=class_labels)
# Now save the model
coreml_model.save('ResNet50.mlmodel')
