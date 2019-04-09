# -*- coding: utf-8 -*-
"""
Author: Chris Dulhanty
"""

import os
import json
from torch.utils.data import Dataset
from skimage import io

CLASS_TO_ID_FILE = 'class_to_id.json'
ID_TO_CLASS_FILE = 'id_to_class.json'
VAL_LABEL_FILE = 'ILSVRC2012_validation_ground_truth.txt'


class ImageNetTrainDataset(Dataset):
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

        with open(ID_TO_CLASS_FILE) as f:
            self.id_to_class_dict = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_folder, os.path.join(self.labels[idx], self.data[idx]))
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        sample = {'image': image,
                  'id': label,
                  'class': self.id_to_class_dict[label]['class'],
                  'string': self.id_to_class_dict[label]['string']}

        return sample


class ImageNetValidationDataset(Dataset):
    def __init__(self, data_folder, transform=None):

        self.data_folder = data_folder

        self.data = [f for f in os.listdir(self.data_folder) if os.path.isfile(os.path.join(self.data_folder, f))]

        self.labels = [img.split('_')[0] for img in self.data]

        assert len(self.data) == len(self.labels)  # ensure the list comprehensions above worked!

        with open(CLASS_TO_ID_FILE) as f:
            self.class_to_id_dict = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_folder, self.data[idx])
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        sample = {'image': image, 'label': label}

        return sample


if __name__ == '__main__':
    train_data_folder = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
    val_data_folder = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/val/'

    train_dataset = ImageNetTrainDataset(train_data_folder)
    val_dataset = ImageNetValidationDataset(val_data_folder)

    print(len(train_dataset))
    item = train_dataset.__getitem__(0)
    print(item['image'].shape, item['id'], item['class'], item['string'])

    print(len(val_dataset))
    item = val_dataset.__getitem__(0)
    print(item['label'], item['image'].shape, item['image'].dtype)
