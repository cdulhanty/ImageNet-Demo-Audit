#!/usr/bin/env python3
"""
Apparent Gender Estimation Model
"""

__author__ = "Chris Dulhanty"
__version__ = "0.1.0"
__license__ = "MIT"

import torch
import imp
import argparse
import json
import os
import numpy as np
import cv2

TRAINING_ROOT = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
GENDER_MODEL_PY_FILE =  '/media/chris/Mammoth/DEX/gender.py'
GENDER_MODEL_PTH_FILE = '/media/chris/Mammoth/DEX/gender.pth'

OUTFILE = 'inference/gender.txt'

GenderModel = imp.load_source('MainModel', GENDER_MODEL_PY_FILE)

DETECTION_THRESHOLD = 0.9  # TODO - determine the optimal level (TP/FP on diff groups?)
FACE_BUFFER = 0.4

WIDTH = 224
HEIGHT = 224
N_CHANNELS = 3
INFERENCE_BATCH_SIZE = 10  # TODO


def expectation(outputs):
    outputs = outputs.detach().numpy().tolist()
    ages = []
    for output in outputs:
        sum = 0
        for age, prob in enumerate(output):
            sum += age*prob
        ages.append(sum)
    return ages


def read_image_batch(batch_list):

    batch_size = len(batch_list)
    image_batch = np.zeros((batch_size, N_CHANNELS, HEIGHT, WIDTH))

    for i, image_details in enumerate(batch_list):

        image = read_image(image_details)
        image_batch[i] = image

    return image_batch


def read_image(image_details):  # loads an image and pre-processes

    image_path = image_details[0]
    details = image_details[1]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_height, img_width, img_channels = img.shape

    # add a buffer to the face crop
    x_add = int(np.round(FACE_BUFFER * details['w']))
    y_add = int(np.round(FACE_BUFFER * details['h']))

    # TODO - add a buffer if the image goes out of range, b/c making it smaller messes up the image when resizing
    xmin = int(np.round(details['xmin'] - x_add)) if int(np.round(details['xmin'] - x_add)) > 0 else 0
    ymin = int(np.round(details['ymin'] - y_add)) if int(np.round(details['ymin'] - y_add)) > 0 else 0
    xmax = xmin + int(np.round(details['w'] + 2*x_add)) if xmin + int(np.round(details['w'] + 2*x_add)) < img_width else img_width
    ymax = ymin + int(np.round(details['h'] + 2*y_add)) if ymin + int(np.round(details['h'] + 2*y_add)) < img_height else img_height

    face = img[ymin:ymax, xmin:xmax].copy()

    # TODO face alignment here - with DLIB?

    face = cv2.resize(face, (HEIGHT, WIDTH))  # TODO resize in a better way (keep the aspect ratio?)
    face = face.astype("float32")
    face -= [104, 117, 124]  # ImageNet mean subtraction
    face = np.moveaxis(face, 2, 0)

    return face


def main(args):

    device = "cuda:0"

    with open('inference/ILSVRC2012_training_dets.json') as f:
        dataset_dict = json.load(f)

    # count the number of valid face detections
    n_valid_dets = 0
    for synset in sorted(dataset_dict['synsets'].keys()):
        for image in sorted(dataset_dict['synsets'][synset]['images'].keys()):
            for face in dataset_dict['synsets'][synset]['images'][image]['faces']:
                if face['score'] > DETECTION_THRESHOLD:
                    n_valid_dets += 1

    # prepare the apparent gender model
    gender_model = torch.load(GENDER_MODEL_PTH_FILE)  # 0 for female, 1 for male
    gender_model.eval()
    gender_model.to(device)

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
                        image_batch = read_image_batch(batch_list)
                        image_batch = torch.from_numpy(image_batch).type(torch.FloatTensor)
                        image_batch = image_batch.to(device)

                        gender_outputs = gender_model(image_batch)

                        print(expectation(gender_outputs.cpu()))
                        return
                        # TODO run the apparent age and gender model here

                        batch_no += 1
                        batch_list = []

                        del image_batch

    if len(batch_list) > 0:
        print('Batch', batch_no, 'of', n_batches)
        image_batch = read_image_batch(batch_list)
        # TODO run the apparent age and gender model here

        batch_no += 1
        batch_list = []

        del image_batch

    print('goodbye')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    # parser.add_argument("arg", help="Required positional argument")

    args = parser.parse_args()
    main(args)
