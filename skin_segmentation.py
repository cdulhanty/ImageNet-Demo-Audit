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

TRAINING_ROOT = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
MODEL_JSON_FILE = 'segmentation_keras_tensorflow/skinseg.json'
MODEL_WEIGHTS_FILE = 'segmentation_keras_tensorflow/skinseg.h5'
OUTFILE = 'inference/ITA.txt'

DETECTION_THRESHOLD = 0.40

WIDTH = 224
HEIGHT = 224
N_CHANNELS = 3
INFERENCE_BATCH_SIZE = 1028


def read_image(loc):  # loads an image and pre-processes

    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (HEIGHT, WIDTH))
    t_image = t_image.astype("float32")
    t_image = keras.applications.densenet.preprocess_input(t_image, data_format='channels_last')

    return t_image


def norm_image(img):  # normalizes and image
    new_img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    return new_img


def read_image_list(flist, start, length, color=1, norm=0):  # loads a set of images from a text index file

    with open(flist) as f:
        content = f.readlines()
    content = [x.strip().split()[0] for x in content]

    datalen = length
    if (datalen < 0):
        datalen = len(content)

    if (start + datalen > len(content)):
        datalen = len(content) - start

    if (color == 1):
        imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, T_G_NUMCHANNELS))
    else:
        imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, 1))

    for i in range(start, start+datalen):
        if ((i-start) < len(content)):
            val = t_read_image(content[i])
            if (color == 0):
                val = val[:,:,0]
                val = np.expand_dims(val,2)
            imgset[i-start] = val
            if (norm == 1):
                imgset[i-start] = (t_norm_image(imgset[i-start]) * 1.0 + 0.0)

    return imgset


def main(args):

    with open('inference/ILSVRC2012_training_dets.json') as f:
        dataset_dict = json.load(f)

    # count the number of valid faces
    n_valid_dets = 0
    for synset in dataset_dict['synsets']:
        for image in synset['images']:
            for face in image['faces']:
                if face['score'] > DETECTION_THRESHOLD:
                    n_valid_dets += 1

    # prepare the skin detection model
    with open(MODEL_JSON_FILE) as f:
        architecture = f.read()
        model = model_from_json(architecture)
    model.load_weights(MODEL_WEIGHTS_FILE, by_name=True)

    n_batches = int(np.ceil(n_valid_dets/float(INFERENCE_BATCH_SIZE)))

    i = 0
    j = 0
    k = 0
    for i in range(n_batches):
        n = 0
        if i < len(dataset_dict['synsets'].keys()):
            if j < len(dataset_dict['synsets']['images'].keys()):
                if k < len(dataset_dict['synsets']['images']['faces']):


                    pass
                k+=1
            j+=1
        i+=1


        imgs = read_image_list(dets_dict.keys(), i * INFERENCE_BATCH_SIZE, INFERENCE_BATCH_SIZE)

        preds = model.predict(imgs)



    for filename in dets_dict.keys():

        filepath = os.path.join(TRAINING_ROOT, filename)

        # open file

        # iterate over the number of images in the detections that are above a threshold (0.4?)

        # run the skin detection

        # convert to CIE Lab space

        # sample x points, calculate the ITA for the face, append value to the dict

    print('goodbye')
    return


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    # parser.add_argument("arg", help="Required positional argument")

    args = parser.parse_args()
    main(args)
