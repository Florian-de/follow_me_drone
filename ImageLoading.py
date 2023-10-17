# imports
import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt
import os

"""
Does not work on Apple Silicon!
# Limit gpu memory growth
gpus = tf.config.experimental.list_logical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(f"GPU is available: {tf.test.is_gpu_available()}")
"""

#images = tf.data.Dataset.list_files(os.path.join("data", "images", "*.jpg") ,shuffle=True)


def load_image(path):
    byte_img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(byte_img)
    return img


"""images = images.map(load_image)

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()

# moved images manually 70/15/15 to train, test, val, now move labels
for folder in ["train", "test", "val"]:
    for file in os.listdir(os.path.join("data", folder, "images")):
        filename = file.split(".")[0]+".json"
        existing_filepath = os.path.join("data", "labels", filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join("data", folder, "labels", filename)
            os.replace(existing_filepath, new_filepath)"""