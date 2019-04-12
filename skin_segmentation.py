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

training_root = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'

def main(args):

    with open('inference/ILSVRC2012_training_dets.json') as f:
        dets_dict = json.load(f)

    # prepare the skin detection model

    for filename in dets_dict.keys():

        filepath = os.path.join(training_root, filename)

        # open file

        # iterate over the number of images in the detections that are above a threshold (0.4?)

        # run the skin detection

        # convert to CIE Lab space

        # sample x points, calculate the ITA for the face, append value to the dict





    """ Main entry point of the app """
    print("hello world")
    print(args)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("arg", help="Required positional argument")

    # Optional argument flag which defaults to False
    parser.add_argument("-f", "--flag", action="store_true", default=False)

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-n", "--name", action="store", dest="name")

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
