#!/usr/bin/env python3
"""
Apparent Gender Estimation Model - Validation on APPA_REAL
"""

__author__ = "Chris Dulhanty"
__version__ = "0.1.0"
__license__ = "MIT"

import torch
import imp
import argparse
import json
import pandas as pd
import os
import numpy as np
import cv2

GENDER_MODEL_PY_FILE = '/media/chris/Mammoth/DEX/gender.py'
GENDER_MODEL_PTH_FILE = '/media/chris/Mammoth/DEX/gender.pth'

DETECTION_JSON = 'inference/APPA_REAL_test_dets.json'
TEST_IMAGE_ROOT = '/media/chris/Mammoth/ChaLearn/appa-real-release/test/'
TEST_LABEL_ROOT = '/media/chris/Mammoth/ChaLearn/allcategories_trainvalidtest_split/allcategories_test.csv'
OUTFILE = 'validation/APPA_REAL_gender.json'

GenderModel = imp.load_source('MainModel', GENDER_MODEL_PY_FILE)

FACE_BUFFER = 0.4
WIDTH = 224
HEIGHT = 224
N_CHANNELS = 3
INFERENCE_BATCH_SIZE = 30


def expectation(outputs):
    outputs = outputs.detach().numpy().tolist()
    preds = []
    for output in outputs:
        sum = 0
        for pred, prob in enumerate(output):
            sum += pred*prob
        preds.append(sum)
    return preds


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
    xmax = xmin + int(np.round(details['w'] + 2 * x_add)) if \
           xmin + int(np.round(details['w'] + 2 * x_add)) < img_width else img_width
    ymax = ymin + int(np.round(details['h'] + 2 * y_add)) if \
           ymin + int(np.round(details['h'] + 2 * y_add)) < img_height else img_height

    face = img[ymin:ymax, xmin:xmax].copy()

    # TODO face alignment here - with DLIB?

    face = cv2.resize(face, (HEIGHT, WIDTH))  # TODO resize in a better way (keep the aspect ratio?)
    face = face.astype("float32")
    face -= [104, 117, 124]  # ImageNet mean subtraction
    face = np.moveaxis(face, 2, 0)

    return face


def main(args):

    device = "cuda:0"

    with open(DETECTION_JSON) as f:
        detection_dict = json.load(f)

    df = pd.read_csv(TEST_LABEL_ROOT)
    df['file'] = df['file'].str.replace(r'.jpg$', '')
    df = df.set_index('file')
    dataset_dict = df.to_dict(orient='index')

    for key in dataset_dict.keys():
        dataset_dict[key]['preds'] = {}

    n_faces = len(detection_dict['images'])

    # prepare the apparent age model
    gender_model = torch.load(GENDER_MODEL_PTH_FILE)  # 0 for female, 1 for male
    gender_model.eval()
    gender_model.to(device)

    n_batches = int(np.ceil(n_faces/float(INFERENCE_BATCH_SIZE)))

    batch_no = 1
    batch_list = []
    n_correct = 0
    for image in sorted(detection_dict['images'].keys()):

        if len(detection_dict['images'][image]['faces']) > 0:
            face = detection_dict['images'][image]['faces'][0]  # TODO better method???

            filepath = os.path.join(TEST_IMAGE_ROOT, image + '.jpg')
            batch_list.append((filepath, face))
        else:
            n_correct -= 1
            print(filepath)

        if len(batch_list) == INFERENCE_BATCH_SIZE:

            print('Batch', batch_no, 'of', n_batches)
            image_batch = read_image_batch(batch_list)
            image_batch = torch.from_numpy(image_batch).type(torch.FloatTensor)
            image_batch = image_batch.to(device)

            gender_outputs = gender_model(image_batch)
            gender_preds = expectation(gender_outputs.cpu())

            for gender_pred, batch_item in zip(gender_preds, batch_list):

                filepath = batch_item[0]

                file_id = filepath.split(TEST_IMAGE_ROOT)[1].split('.')[0]
                appa_gender_str = dataset_dict[file_id]['gender']
                appa_gender = 0 if appa_gender_str == 'female' else 1
                gender_pred = 0 if gender_pred < 0.50 else 1

                if appa_gender == gender_pred:
                    n_correct += 1
                dataset_dict[file_id]['preds']['DEX'] = {'pred': gender_pred}

            batch_no += 1
            batch_list = []

            del image_batch

    if len(batch_list) > 0:

        print('Batch', batch_no, 'of', n_batches)
        image_batch = read_image_batch(batch_list)
        image_batch = torch.from_numpy(image_batch).type(torch.FloatTensor)
        image_batch = image_batch.to(device)

        gender_outputs = gender_model(image_batch)
        gender_preds = expectation(gender_outputs.cpu())

        for gender_pred, batch_item in zip(gender_preds, batch_list):

            filepath = batch_item[0]

            file_id = filepath.split(TEST_IMAGE_ROOT)[1].split('.')[0]
            appa_gender_str = dataset_dict[file_id]['gender']
            appa_gender = 0 if appa_gender_str == 'female' else 1
            gender_pred = 0 if gender_pred < 0.50 else 1

            if appa_gender == gender_pred:
                n_correct += 1
            dataset_dict[file_id]['preds']['DEX'] = {'pred': gender_pred}

        del image_batch

    print('Accuracy:', n_correct/float(n_faces))

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
