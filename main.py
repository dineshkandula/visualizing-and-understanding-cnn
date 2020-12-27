import numpy as np

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications import vgg16

from PIL import Image
import matplotlib.pyplot as plt

from visualize_cnn import visualize

TEST_IMAGE_PATH = 'test_images/car.jpeg'

VISUALIZE_LAYER = 'block3_conv3'
VISUALIZE_FEATURE = 127
VISUALIZE_MODE='all' #or 'max'

vgg_model = vgg16.VGG16(weights='imagenet', include_top = True)

img = load_img(TEST_IMAGE_PATH, target_size=(224, 224), interpolation='nearest')
numpy_image = img_to_array(img)
image_batch = np.expand_dims(numpy_image, axis=0)
numpy_image = numpy_image.astype(np.float)
# prepare the image for the VGG model
img = vgg16.preprocess_input(image_batch)

plt.imshow(img[0])
plt.show()

visualize(img, vgg_model, VISUALIZE_LAYER, VISUALIZE_FEATURE, VISUALIZE_MODE)
