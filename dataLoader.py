# -*- coding: utf-8 -*-
"""
Author: Chris Dulhanty
"""

import os
import json
from torch.utils.data import Dataset
from skimage import io

ID_TO_CLASS_FILE = 'id_to_class.json'
CLASS_TO_ID_FILE = 'class_to_id.json'

VAL_LABEL_FILE = 'imagenet_2012_validation_synset_labels.txt'


class ImageNetTrainDataset(Dataset):
    def __init__(self, data_folder, transform=None):

        self.data_folder = data_folder

        self.data = []
        self.labels = []
        sub_dirs = sorted([sub_dir for sub_dir in os.listdir(self.data_folder) if os.path.isdir(os.path.join(self.data_folder, sub_dir))])

        for sub_dir in sub_dirs:
            joined_sub_dir = os.path.join(self.data_folder, sub_dir)
            files = sorted([f for f in os.listdir(joined_sub_dir) if os.path.isfile(os.path.join(joined_sub_dir, f))])
            for file in files:
                self.data.append(file)
                self.labels.append(sub_dir)

        assert len(self.data) == len(self.labels)  # ensure the list comprehensions above worked!

        with open(ID_TO_CLASS_FILE) as f:
            self.id_to_class_dict = json.load(f)

        with open(CLASS_TO_ID_FILE) as f:
            self.class_to_id_dict = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_folder, os.path.join(self.labels[idx], self.data[idx]))
        image = io.imread(img_path)

        if self.transform:  # TODO - format the data for best practices
            image = self.transform(image)

        label = self.labels[idx]

        sample = {'image': image,
                  'id': label,
                  'class': self.id_to_class_dict[label]['class'],
                  'string': self.id_to_class_dict[label]['string'],
                  'filename': self.data[idx]}

        return sample


class ImageNetValidationDataset(Dataset):
    def __init__(self, data_folder, transform=None):

        self.data_folder = data_folder

        self.data = sorted([f for f in os.listdir(self.data_folder) if os.path.isfile(os.path.join(self.data_folder, f))])

        self.labels = []
        with open(VAL_LABEL_FILE) as f:
            for line in f:
                self.labels.append(line.strip())

        assert len(self.data) == len(self.labels)  # ensure the list comprehension above worked!

        with open(ID_TO_CLASS_FILE) as f:
            self.id_to_class_dict = json.load(f)

        with open(CLASS_TO_ID_FILE) as f:
            self.class_to_id_dict = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.data_folder, self.data[idx])
        image = io.imread(img_path)

        if self.transform:  # TODO - format the data for best practices
            image = self.transform(image)

        label = self.labels[idx]

        sample = {'image': image,
                  'id': label,
                  'class': self.id_to_class_dict[label]['class'],
                  'string': self.id_to_class_dict[label]['string'],
                  'filename': self.data[idx]}

        return sample


if __name__ == '__main__':
    train_data_folder = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
    val_data_folder = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/val/'

    train_dataset = ImageNetTrainDataset(train_data_folder)
    val_dataset = ImageNetValidationDataset(val_data_folder)

    print(len(train_dataset))
    train_item = train_dataset.__getitem__(1)
    print(train_item['image'].shape, train_item['id'], train_item['class'], train_item['string'], train_item['filename'])

    print(len(val_dataset))
    val_item = val_dataset.__getitem__(1)
    print(val_item['image'].shape, val_item['id'], val_item['class'], val_item['string'], val_item['filename'])
