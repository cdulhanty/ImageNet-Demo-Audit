#!/usr/bin/env python3
"""
ImageNet Demographics Audit
"""

__author__ = "Chris Dulhanty"
__version__ = "0.1.0"
__license__ = "MIT"

import torch
import argparse
from torchvision import transforms
from dataLoader import ImageNetTrainDataset


def main(main_args):

    batch_size = 1

    image_size = 300

    # instantiate data loaders
    train_transforms = transforms.Compose([
        # transforms.Resize([image_size, image_size]),
        transforms.ToTensor()
    ])

    train_dataset = ImageNetTrainDataset(main_args.root_dir, transform=train_transforms)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8)

    # run face detection model

    # extract face crops

    # apply DEX model for apparent gender and age prediction

    # apply skin mask model

    # convert image to CIE-Lab space, randomly sample n pixels (or do them all?), point-wise calculation of ITA
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="root directory of dataset to be annotated")
    args = parser.parse_args()
    main(args)
