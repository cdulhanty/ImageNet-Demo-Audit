#!/usr/bin/env python3
"""
Apparent Gender Estimation Model - Validation on ChaLearn
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

TEST_IMAGE_ROOT = '/media/chris/Mammoth/ChaLearn/appa-real-release/test/'
TEST_LABEL_ROOT = '/media/chris/Mammoth/ChaLearn/allcategories_trainvalidtest_split/allcategories_test.csv'
OUTFILE = 'validation/gender.json'

GenderModel = imp.load_source('MainModel', GENDER_MODEL_PY_FILE)

WIDTH = 224
HEIGHT = 224
N_CHANNELS = 3
INFERENCE_BATCH_SIZE = 24


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


def read_image(image_path):  # loads an image and pre-processes

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (HEIGHT, WIDTH))
    img = img.astype("float32")
    img -= [104, 117, 124]  # ImageNet mean subtraction
    img = np.moveaxis(img, 2, 0)

    return img


def main(args):

    device = "cuda:0"

    file_list = sorted([f for f in os.listdir(TEST_IMAGE_ROOT) if
                        os.path.isfile(os.path.join(TEST_IMAGE_ROOT, f)) and '_face' in f])

    df = pd.read_csv(TEST_LABEL_ROOT)
    df['file'] = df['file'].str.replace(r'.jpg$', '')
    df = df.set_index('file')
    dataset_dict = df.to_dict(orient='index')

    for key in dataset_dict.keys():
        dataset_dict[key]['preds'] = {}

    n_faces = len(file_list)

    # prepare the apparent age model -> 0 for female, 1 for male
    gender_model = torch.load(GENDER_MODEL_PTH_FILE)
    gender_model.eval()
    gender_model.to(device)

    n_batches = int(np.ceil(n_faces/float(INFERENCE_BATCH_SIZE)))

    batch_no = 1
    batch_list = []
    n_correct = 0
    for file in file_list:

        filepath = os.path.join(TEST_IMAGE_ROOT, file)
        batch_list.append(filepath)

        if len(batch_list) == INFERENCE_BATCH_SIZE:

            print('Batch', batch_no, 'of', n_batches)
            image_batch = read_image_batch(batch_list)
            image_batch = torch.from_numpy(image_batch).type(torch.FloatTensor)
            image_batch = image_batch.to(device)

            gender_outputs = gender_model(image_batch)
            gender_preds = expectation(gender_outputs.cpu())

            for gender_pred, filepath in zip(gender_preds, batch_list):

                file_id = filepath.split(TEST_IMAGE_ROOT)[1].split('.')[0]
                appa_gender_str = dataset_dict[file_id]['gender']
                appa_gender = 0 if appa_gender_str == 'female' else 1
                gender_pred = 0 if gender_pred < 0.50 else 1

                if appa_gender == gender_pred:
                    n_correct += 1
                else:
                    cv2.imshow(filepath, cv2.imread(filepath, cv2.IMREAD_COLOR))

                dataset_dict[file_id]['preds']['DEX'] = {'pred': gender_pred}
            cv2.waitKey(0)
            cv2.destroyAllWindows()
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

        for gender_pred, filepath in zip(gender_preds, batch_list):

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
