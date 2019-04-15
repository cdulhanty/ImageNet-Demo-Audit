#!/usr/bin/env python3
"""
Skin Segmentation Model and ITA Calculation
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

MODEL_JSON_FILE = 'segmentation_keras_tensorflow/skinseg.json'
MODEL_WEIGHTS_FILE = 'segmentation_keras_tensorflow/skinseg.h5'

DETECTION_JSON = 'inference/ILSVRC2012_training_dets.json'
TRAINING_ROOT = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
OUTFILE = 'inference/ILSVRC2012_training_skin_type.json'

SKIN_MASK_THRESHOLD = 0.999  # TODO - determine the optimal value (ECU dataset?)
DETECTION_THRESHOLD = 0.9  # TODO - determine the optimal level (TP/FP on diff groups?)
FACE_BUFFER = 0.4

WIDTH = 224
HEIGHT = 224
N_CHANNELS = 3
INFERENCE_BATCH_SIZE = 64


def ITA(L, b):
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
    x_add = int(np.round(FACE_BUFFER * details['w']))
    y_add = int(np.round(FACE_BUFFER * details['h']))

    # TODO - add a buffer if the image goes out of range, b/c making it smaller messes up the image when resizing
    xmin = int(np.round(details['xmin'] - x_add)) if int(np.round(details['xmin'] - x_add)) > 0 else 0
    ymin = int(np.round(details['ymin'] - y_add)) if int(np.round(details['ymin'] - y_add)) > 0 else 0
    xmax = xmin + int(np.round(details['w'] + 2 * x_add)) if \
           xmin + int(np.round(details['w'] + 2 * x_add)) < img_width else img_width
    ymax = ymin + int(np.round(details['h'] + 2 * y_add)) if \
           ymin + int(np.round(details['h'] + 2 * y_add)) < img_height else img_height

    face = img[ymin:ymax, xmin:xmax].copy()

    cie_lab_face = face.copy()

    # TODO face alignment here - with DLIB?
    face = cv2.resize(face, (HEIGHT, WIDTH))  # TODO resize in a better way (keep the aspect ratio?)
    face = face.astype("float32")
    face = preprocess_input(face, data_format='channels_last')

    cie_lab_face = cv2.cvtColor(cie_lab_face, cv2.COLOR_RGB2LAB)  # convert to the CIE L*A*B colorspace
    cie_lab_face = cv2.resize(cie_lab_face, (HEIGHT, WIDTH))  #
    cie_lab_face = cie_lab_face.astype("float32")

    return face, cie_lab_face


def main(args):

    with open(DETECTION_JSON) as f:
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

                if face['score'] > DETECTION_THRESHOLD:  # TODO - should we reject any small faces? ('w' or 'h' < 20)

                    filepath = os.path.join(TRAINING_ROOT, os.path.join(synset, image))
                    batch_list.append((filepath, face))

                    if len(batch_list) == INFERENCE_BATCH_SIZE:

                        print('Batch', batch_no, 'of', n_batches)
                        image_batch, cie_lab_image_batch = read_image_batch(batch_list)
                        skin_masks = model.predict(image_batch)  # run the skin segmentation model

                        for skin_mask, cie_lab_image, batch_item in zip(skin_masks, cie_lab_image_batch, batch_list):

                            """
                            for threshold in [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
                                mask_vector = skin_mask.flatten()
                                mask_vector[mask_vector >= threshold] = 1
                                mask_vector[mask_vector < threshold] = 0

                                # extract mean L and b values for input into ITA calculation
                                l_vector = cie_lab_image[:, :, 0].flatten()
                                l_vector_masked = l_vector * mask_vector

                                l_values = l_vector_masked[l_vector_masked > 0]

                                if len(l_values) > 0:

                                    l_mean = np.mean(l_values)

                                    b_vector = cie_lab_image[:, :, 2].flatten()
                                    b_vector_masked = b_vector * mask_vector
                                    b_values = b_vector_masked[b_vector_masked > 0]
                                    b_mean = np.mean(b_values)

                                    ita = ITA(l_mean, b_mean)

                                    print(threshold, len(l_values), ita)
                                else:
                                    print(threshold, '0', '???')
                            """
                            # Threshold the skin mask @ the designated threshold
                            mask_vector = skin_mask.flatten()

                            mask_vector[mask_vector >= SKIN_MASK_THRESHOLD] = 1
                            mask_vector[mask_vector < SKIN_MASK_THRESHOLD] = 0

                            # extract mean L and b values for input into ITA calculation
                            l_vector = cie_lab_image[:, :, 0].flatten()
                            l_vector_masked = l_vector * mask_vector

                            l_values = l_vector_masked[l_vector_masked > 0]
                            l_mean = np.mean(l_values)

                            b_vector = cie_lab_image[:, :, 2].flatten()
                            b_vector_masked = b_vector * mask_vector
                            b_values = b_vector_masked[b_vector_masked > 0]
                            b_mean = np.mean(b_values)

                            ita = ITA(l_mean, b_mean)

                            filepath = batch_item[0]
                            synset_filename = filepath.split(TRAINING_ROOT)[1]
                            synset, filename = synset_filename.split('/')

                            if 'ITA' not in dataset_dict['synsets'][synset]['images'][filename]:
                                dataset_dict['synsets'][synset]['images'][filename]['ITA'] = []

                            dataset_dict['synsets'][synset]['images'][filename]['ITA'].append(ita)

                        batch_no += 1
                        batch_list = []

                        del image_batch
                        del cie_lab_image_batch
                        del skin_masks

    if len(batch_list) > 0:

        print('Batch', batch_no, 'of', n_batches)
        image_batch, cie_lab_image_batch = read_image_batch(batch_list)
        skin_masks = model.predict(image_batch)  # run the skin segmentation model

        for skin_mask, cie_lab_image, batch_item in zip(skin_masks, cie_lab_image_batch, batch_list):

            # Threshold the skin mask @ the designated threshold
            mask_vector = skin_mask.flatten()

            mask_vector[mask_vector >= SKIN_MASK_THRESHOLD] = 1
            mask_vector[mask_vector < SKIN_MASK_THRESHOLD] = 0

            # extract mean L and b values for input into ITA calculation
            l_vector = cie_lab_image[:, :, 0].flatten()
            l_vector_masked = l_vector * mask_vector

            l_values = l_vector_masked[l_vector_masked > 0]
            l_mean = np.mean(l_values)

            b_vector = cie_lab_image[:, :, 2].flatten()
            b_vector_masked = b_vector * mask_vector
            b_values = b_vector_masked[b_vector_masked > 0]
            b_mean = np.mean(b_values)

            ita = ITA(l_mean, b_mean)

            filepath = batch_item[0]
            synset_filename = filepath.split(TRAINING_ROOT)[1]
            synset, filename = synset_filename.split('/')

            if 'ITA' not in dataset_dict['synsets'][synset]['images'][filename]:
                dataset_dict['synsets'][synset]['images'][filename]['ITA'] = []

            dataset_dict['synsets'][synset]['images'][filename]['ITA'].append(ita)

        del image_batch
        del cie_lab_image_batch
        del skin_masks

    out_json = json.dumps(dataset_dict)
    with open(OUTFILE, 'w') as f:
        f.write(out_json)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional argument
    # parser.add_argument("arg", help="Required positional argument")

    args = parser.parse_args()
    main(args)
