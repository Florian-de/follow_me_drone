import json
import os

import cv2
import numpy as np
import tensorflow as tf
from ImageLoading import load_image
from matplotlib import pyplot as plt


def load_images_to_tf(images_path):
    """
    Loads the images into a tf datasets, resizes them to 120x120 and normalize the rgb values
    :param path: path to all the images with wildcard
    :return: the images loaded in a tf dataset
    """
    images = tf.data.Dataset.list_files(images_path)
    images = images.map(load_image)
    images = images.map(lambda x: tf.image.resize(x, (120, 120)))
    images = images.map(lambda x: x / 255)
    return images


def _load_labels(label_path):
    """
    Extracts class and bounding box from label jsons
    :param label_path: path to the label jsons
    :return: class and bounding box
    """
    with open(label_path.numpy(), "r", encoding="utf-8") as f:
        label = json.load(f)
    return [label["class"]], label["bbox"]


def load_labels_to_tf(labels_path):
    """
    Loads the labels into a tf dataset
    :param labels_path: path to the labels
    :return: tf dataset with the labels
    """
    labels = tf.data.Dataset.list_files(labels_path, shuffle=False)
    labels = labels.map(lambda x: tf.py_function(_load_labels, [x], [tf.uint8, tf.float16]))
    return labels


def create_shuffled_dataset(images, labels):
    """
    Builds combined, shuffled, batched and prefetched dataset
    :param images: image dataset
    :param labels: labels dataset
    :return: builded dataset
    """
    data = tf.data.Dataset.zip(images, labels)
    data = data.shuffle(5000)
    data = data.batch(8)
    data = data.prefetch(4)
    return data


train_images = load_images_to_tf(os.path.join("aug_data", "train", "images", "*.jpg"))
test_images = load_images_to_tf(os.path.join("aug_data", "test", "images", "*.jpg"))
val_images = load_images_to_tf(os.path.join("aug_data", "val", "images", "*.jpg"))

train_labels = load_labels_to_tf(os.path.join("aug_data", "train", "labels", "*.json"))
test_labels = load_labels_to_tf(os.path.join("aug_data", "test", "labels", "*.json"))
val_labels = load_labels_to_tf(os.path.join("aug_data", "val", "labels", "*.json"))

# Check if there are the same amount of images and labels
print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))

train = create_shuffled_dataset(train_images, train_labels)
test = create_shuffled_dataset(test_images, test_labels)
val = create_shuffled_dataset(val_images, val_labels)

# show batches for testing purposes
res = train.as_numpy_iterator().next()
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                  (255, 0, 0), 2)

    ax[idx].imshow(sample_image)