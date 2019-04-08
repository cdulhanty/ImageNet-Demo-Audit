# -*- coding: utf-8 -*-
"""
Author: Chris Dulhanty
"""

import os
from torch.utils.data import Dataset
from skimage import io

class ImageNetDataset(Dataset):
    def __init__(self, data_folder, transform=None):

        self.data_folder = data_folder

        self.data = [f for sub_dir in os.listdir(self.data_folder) for f in
                     os.listdir(os.path.join(self.data_folder, sub_dir)) if
                     os.path.isfile(os.path.join(self.data_folder, (os.path.join(sub_dir, f))))]

        self.labels = [sub_dir for sub_dir in os.listdir(self.data_folder) for _ in
                       range(len([f for f in os.listdir(os.path.join(self.data_folder, sub_dir)) if
                       os.path.isfile(os.path.join(self.data_folder, (os.path.join(sub_dir, f))))])) if
                       os.path.isdir(os.path.join(self.data_folder, sub_dir))]

        assert len(self.data) == len(self.labels)  # ensure the list comprehensions above worked!

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_folder, os.path.join(self.labels[idx], self.data[idx]))
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


if __name__ == '__main__':
    train_data_folder = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
    # val_data_folder = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/val/'  TODO fix

    train_dataset = ImageNetDataset(data_folder=train_data_folder)
    # val_dataset = ImageNetDataset(data_folder=val_data_folder)

    print(len(train_dataset))
    # print(len(val_dataset))

    image, label = train_dataset.__getitem__(0)
    print(label, image.shape)
