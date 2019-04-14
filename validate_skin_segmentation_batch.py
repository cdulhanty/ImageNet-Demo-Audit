#!/usr/bin/env python3
"""
Validate Skin Segmentation Model
"""

__author__ = "Chris Dulhanty"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
import json
import os
from keras.models import model_from_json
import numpy as np
import cv2
from keras.applications.densenet import preprocess_input
from sklearn.metrics import f1_score, jaccard_similarity_score, precision_recall_curve, accuracy_score

MODEL_JSON_FILE = 'segmentation_keras_tensorflow/skinseg.json'
MODEL_WEIGHTS_FILE = 'segmentation_keras_tensorflow/skinseg.h5'

EXAMPLES_ROOT = '/media/chris/Mammoth/FSD/Original/'
MASKS_ROOT = '/media/chris/Mammoth/FSD/Skin/'

OUTFILE = 'validation/skin_segmentation.json'

WIDTH = 224
HEIGHT = 224
N_CHANNELS = 3

# TODO - create a function to create a beginning JSON object for the datasets that are being evaluated (diff file)


def segmentation_metrics(y_true, y_pred):

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred, pos_label=1)

    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_threshold = 0

    for p, r, t in zip(precision, recall, thresholds):
        f1 = 2 * (p * r) / (p + r)

        if f1 > best_f1:
            best_precision = p
            best_recall = r
            best_threshold = t
            best_f1 = f1

    # threshold y_pred at the best threshold
    y_pred[y_pred >= best_threshold] = 1
    y_pred[y_pred < best_threshold] = 0

    # run accuracy, jaccard, f1 measures
    accuracy = accuracy_score(y_true, y_pred)
    jaccard = jaccard_similarity_score(y_true, y_pred)

    return best_precision, best_recall, best_f1, accuracy, jaccard, best_threshold


def main(args):

    # prepare the skin detection model
    with open(MODEL_JSON_FILE) as f:
        architecture = f.read()
        model = model_from_json(architecture)
    model.load_weights(MODEL_WEIGHTS_FILE, by_name=True)

    dataset_dict = {}

    example_filenames = sorted([f for f in os.listdir(EXAMPLES_ROOT) if os.path.isfile(os.path.join(EXAMPLES_ROOT, f))])
    mask_filenames = sorted([f for f in os.listdir(MASKS_ROOT) if os.path.isfile(os.path.join(MASKS_ROOT, f))])

    # TODO Train / Test Split? (Is there a standard split?)
    y_pred = []
    y_true = []

    for example_no, (example_filename, mask_filename) in enumerate(zip(example_filenames, mask_filenames)):

        if example_no % 100 == 0:
            print(example_no)

        example_filepath = os.path.join(EXAMPLES_ROOT, example_filename)  # load the image file
        image = cv2.imread(example_filepath, cv2.IMREAD_COLOR)

        img_width, img_height, img_channels = image.shape

        image = cv2.resize(image, (HEIGHT, WIDTH))  # TODO resize in a better way (keep the aspect ratio?)
        image = image.astype("float32")
        image = preprocess_input(image, data_format='channels_last')
        image = np.expand_dims(image, axis=0)

        pred_mask = model.predict(image)[0]  # run the segmentation mask model
        pred_mask = cv2.resize(pred_mask, (img_height, img_width))  # resize to the original dimensions
        y_pred.append(pred_mask.flatten())

        mask_filepath = os.path.join(MASKS_ROOT, mask_filename)  # load the ground truth mask
        mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype("float32")
        mask /= 255  # make binary
        mask = 1 - mask  # flip

        y_true.append(mask.flatten())

        if example_no >= 500:
            break

    precision, recall, f1, accuracy, jaccard, threshold = segmentation_metrics(y_true, y_pred)

    print('precision', precision, 'recall', recall, 'f1', f1, 'acc', accuracy, 'jaccard', jaccard, 'threshold', threshold)

    """
    image_dict = {}
    image_dict['height'] = img_height
    image_dict['width'] = img_width
    image_dict['model_results'] = {}
    image_dict['model_results']['CodellaPreTrained'] = {}
    image_dict['model_results']['CodellaPreTrained']['precision'] = precision
    image_dict['model_results']['CodellaPreTrained']['recall'] = recall
    image_dict['model_results']['CodellaPreTrained']['f1'] = f1a
    image_dict['model_results']['CodellaPreTrained']['jaccard'] = jaccard

    dataset_dict[example_filename] = image_dict
    """

    # log to a json file TODO

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required positional argument
    # parser.add_argument("arg", help="Required positional argument")
    args = parser.parse_args()
    main(args)
