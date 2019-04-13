#!/usr/bin/env python3
"""
Skin Segmentation and ITA Calculation Model
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

TRAINING_ROOT = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
MODEL_JSON_FILE = 'segmentation_keras_tensorflow/skinseg.json'
MODEL_WEIGHTS_FILE = 'segmentation_keras_tensorflow/skinseg.h5'
OUTFILE = 'inference/ITA.txt'

DETECTION_THRESHOLD = 0.9  # TODO - determine the optimal level (TP/FP on diff groups?)

WIDTH = 224
HEIGHT = 224
N_CHANNELS = 3
INFERENCE_BATCH_SIZE = 512


def ITA(pixel):

    # TODO - might actually pass in something like (1, 1, 1) - need to extract the correct value from this

    L = pixel[0]
    a = pixel[1]
    b = pixel[2]

    return (np.arctan((L - 50)/b) * 180)/np.pi


def read_image_batch(batch_list):

    batch_size = len(batch_list)
    image_batch = np.zeros((batch_size, HEIGHT, WIDTH, N_CHANNELS))
    cie_lab_image_batch = np.zeros((batch_size, HEIGHT, WIDTH, N_CHANNELS))

    for i, image_details in enumerate(batch_list):

        image, cie_lab_image = read_image(image_details)
        image_batch[i] = image
        cie_lab_image_batch[i] = cie_lab_image

    return image_batch, cie_lab_image_batch


def read_image(image_details):  # loads an image and pre-processes

    image_path = image_details[0]
    details = image_details[1]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_height, img_width, img_channels = img.shape

    # add a 50% buffer to the face crop
    x_add = int(np.round(0.5 * details['w']))
    y_add = int(np.round(0.5 * details['h']))

    # TODO - add a buffer if the image goes out of range, b/c making it smaller messes up the image when resizing
    xmin = int(np.round(details['xmin'] - x_add)) if int(np.round(details['xmin'] - x_add)) > 0 else 0
    ymin = int(np.round(details['ymin'] - y_add)) if int(np.round(details['ymin'] - y_add)) > 0 else 0
    xmax = xmin + int(np.round(details['w'] + 2*x_add)) if xmin + int(np.round(details['w'] + 2*x_add)) < img_width else img_width
    ymax = ymin + int(np.round(details['h'] + 2*y_add)) if ymin + int(np.round(details['h'] + 2*y_add)) < img_height else img_height

    face = img[ymin:ymax, xmin:xmax].copy()
    cie_lab_face = face.copy()

    face = cv2.resize(face, (HEIGHT, WIDTH))  # TODO resize in a better way (keep the aspect ratio?)
    face = face.astype("float32")
    face = preprocess_input(face, data_format='channels_last')

    cie_lab_face = cv2.cvtColor(cie_lab_face, cv2.COLOR_RGB2LAB)
    cie_lab_face = cv2.resize(cie_lab_face, (HEIGHT, WIDTH))
    cie_lab_face = cie_lab_face.astype("float32")

    return face, cie_lab_face


def main(args):

    with open('inference/ILSVRC2012_training_dets.json') as f:
        dataset_dict = json.load(f)

    # count the number of valid face detections
    n_valid_dets = 0
    for synset in sorted(dataset_dict['synsets'].keys()):
        for image in sorted(dataset_dict['synsets'][synset]['images'].keys()):
            for face in dataset_dict['synsets'][synset]['images'][image]['faces']:
                if face['score'] > DETECTION_THRESHOLD:
                    n_valid_dets += 1

    # prepare the skin detection model
    with open(MODEL_JSON_FILE) as f:
        architecture = f.read()
        model = model_from_json(architecture)
    model.load_weights(MODEL_WEIGHTS_FILE, by_name=True)

    n_batches = int(np.ceil(n_valid_dets/float(INFERENCE_BATCH_SIZE)))

    batch_no = 1
    batch_list = []
    for synset in sorted(dataset_dict['synsets'].keys()):
        for image in sorted(dataset_dict['synsets'][synset]['images'].keys()):
            for face in dataset_dict['synsets'][synset]['images'][image]['faces']:

                if face['score'] > DETECTION_THRESHOLD:  # TODO - should we reject any small faces ('w' < 20 or 'h' < 20)??

                    filepath = os.path.join(TRAINING_ROOT, os.path.join(synset, image))
                    batch_list.append((filepath, face))

                    if len(batch_list) == INFERENCE_BATCH_SIZE:

                        print('Batch', batch_no, 'of', n_batches)
                        image_batch, cie_lab_image_batch = read_image_batch(batch_list)
                        skin_masks = model.predict(image_batch)  # run the skin segmentation model

                        for skin_mask, cie_lab_image in zip(skin_masks, cie_lab_image_batch):

                            # TODO - run the dlib function?
                            # TODO - determine how to select the pixels to extract, how to filter, and pass to ITA()
                            # TODO - add calculated ITA value to the dataset_dict
                            pass

                        batch_no += 1
                        batch_list = []

                        del image_batch
                        del cie_lab_image_batch
                        del skin_masks

    if len(batch_list) > 0:

        print('Batch', batch_no, 'of', n_batches)
        image_batch, cie_lab_image_batch = read_image_batch(batch_list)
        skin_masks = model.predict(image_batch)  # run the skin segmentation model

        for skin_mask, cie_lab_image in zip(skin_masks, cie_lab_image_batch):
            # TODO - run the dlib function?
            # TODO - determine how to select the pixels to extract, how to filter, and pass to ITA()
            # TODO - add calculated ITA value to the dataset_dict
            pass

        del image_batch
        del cie_lab_image_batch
        del skin_masks

    print('goodbye')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    # parser.add_argument("arg", help="Required positional argument")

    args = parser.parse_args()
    main(args)
