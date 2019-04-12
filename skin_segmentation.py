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

TRAINING_ROOT = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
MODEL_JSON_FILE = 'segmentation_keras_tensorflow/skinseg.json'
MODEL_WEIGHTS_FILE = 'segmentation_keras_tensorflow/skinseg.h5'


def main(args):

    with open('inference/ILSVRC2012_training_dets.json') as f:
        dets_dict = json.load(f)

    # prepare the skin detection model

    with open(MODEL_JSON_FILE) as f:
        architecture = f.read()
        model = model_from_json(architecture)

    # model = model_from_json(MODEL_JSON_FILE)
    model.load_weights(MODEL_WEIGHTS_FILE, by_name=True)
    model.summary()

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
