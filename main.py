#!/usr/bin/env python3
"""
ImageNet Demographics Audit
"""

__author__ = "Chris Dulhanty"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse


def main(args):

    print("hello world")
    print(args)

    # instantiate data loaders

    # run face detection model

    # extract face crops

    # apply DEX model for apparent gender and age prediction

    # apply skin mask model

    # convert image to CIE-Lab space, randomly sample n pixels (or do them all?), point-wise calculation of ITA


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
