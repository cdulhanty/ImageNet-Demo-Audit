#!/usr/bin/env python3
"""
ImageNet Face Detection
"""

__author__ = "Chris Dulhanty"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import cv2
import torch
import argparse
import numpy as np
import json
from FaceBoxes.models.faceboxes import FaceBoxes
import torch.backends.cudnn as cudnn
from FaceBoxes.layers.functions.prior_box import PriorBox
from FaceBoxes.utils.box_utils import decode
from FaceBoxes.data import cfg
from FaceBoxes.utils.nms_wrapper import nms

ROOT_DIR = '/media/chris/Datasets/ILSVRC/imagenet_object_localization/ILSVRC/Data/CLS-LOC/train/'
ID_TO_CLASS_FILE = 'id_to_class.json'
OUTFILE = 'inference/ILSVRC2012_training_dets'

FACEBOXES_WEIGHTS_PATH = 'FaceBoxes/weights/FaceBoxes.pth'
RESIZE = 1.0
CONFIDENCE_THRESHOLD = 0.050
NMS_THREHOLD = 0.3
TOP_K = 5000
KEEP_TOP_K = 750


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path):
    print('Loading pretrained FaceBoxes model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def main(args):

    data = []
    labels = []
    sub_dirs = sorted([sub_dir for sub_dir in os.listdir(ROOT_DIR) if
                       os.path.isdir(os.path.join(ROOT_DIR, sub_dir))])

    for sub_dir in sub_dirs:
        joined_sub_dir = os.path.join(ROOT_DIR, sub_dir)
        files = sorted([f for f in os.listdir(joined_sub_dir) if os.path.isfile(os.path.join(joined_sub_dir, f))])
        for file in files:
            data.append(file)
            labels.append(sub_dir)

    with open(ID_TO_CLASS_FILE) as f:
        id_to_class_dict = json.load(f)

    # Face Detection Model - FaceBox
    torch.set_grad_enabled(False)
    net = FaceBoxes(phase='test', size=None, num_classes=2)  # initialize detector
    net = load_model(net, FACEBOXES_WEIGHTS_PATH)
    net.eval()
    cudnn.benchmark = True
    device = torch.device("cuda")
    net = net.to(device)

    fw = open(OUTFILE + '.txt', 'w')

    # Inference
    for i, filename in enumerate(data):
        if i % 1000 == 0:
            print(i)

        synset_id = labels[i]

        img_path = os.path.join(ROOT_DIR, os.path.join(synset_id, filename))
        img = np.float32(cv2.imread(img_path, cv2.IMREAD_COLOR))

        if RESIZE != 1:
            img = cv2.resize(img, None, None, fx=RESIZE, fy=RESIZE, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = img.shape

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)

        img = img.to(device)

        scale = scale.to(device)

        out = net(img)  # forward pass
        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(device)
        loc, conf, _ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / RESIZE
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > CONFIDENCE_THRESHOLD)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:TOP_K]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(dets, NMS_THREHOLD)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:KEEP_TOP_K, :]

        # inference dets
        fw.write('{:s}\n'.format(filename))
        fw.write('{:.1f}\n'.format(dets.shape[0]))
        for k in range(dets.shape[0]):
            xmin = dets[k, 0]
            ymin = dets[k, 1]
            xmax = dets[k, 2]
            ymax = dets[k, 3]
            score = dets[k, 4]
            w = xmax - xmin + 1
            h = ymax - ymin + 1
            fw.write('{:.3f} {:.3f} {:.3f} {:.3f} {:.10f}\n'.format(xmin, ymin, w, h, score))

    fw.close()

    dataset_dict = {}
    dataset_dict['n_faces'] = 0
    dataset_dict['synsets'] = {}

    lines = []

    with open(OUTFILE + '.txt') as f:

        for i, l in enumerate(f):
            lines.append(l)
        n_lines = i + 1

        j = 0
        while j < n_lines:

            if '.JPEG' in lines[j]:
                image_dict = {}
                image_dict['n_faces'] = 0
                image_dict['faces'] = []

                synset = lines[j].split('_')[0]

                if synset not in dataset_dict['synsets']:
                    dataset_dict['synsets'][synset] = {}
                    dataset_dict['synsets'][synset]['n_faces'] = 0
                    dataset_dict['synsets'][synset]['images'] = {}
                    dataset_dict['synsets'][synset]['class'] = id_to_class_dict[synset]['class']
                    dataset_dict['synsets'][synset]['string'] = id_to_class_dict[synset]['string']

            if is_float(lines[j + 1]):

                n_detections = int(float(lines[j + 1]))

                for k in range(n_detections):
                    detection_list = lines[j + 2 + k].split()
                    detection_dict = {}

                    detection_dict['xmin'] = float(detection_list[0])
                    detection_dict['ymin'] = float(detection_list[1])
                    detection_dict['w'] = float(detection_list[2])
                    detection_dict['h'] = float(detection_list[3])
                    detection_dict['score'] = float(detection_list[4])

                    image_dict['faces'].append(detection_dict)
                    image_dict['n_faces'] += 1

                dataset_dict['synsets'][synset]['n_faces'] += image_dict['n_faces']
                dataset_dict['synsets'][synset]['images'][lines[j].strip()] = image_dict
                dataset_dict['n_faces'] += image_dict['n_faces']

                j += n_detections + 2

    with open(OUTFILE + '.json', 'w') as f:
        json.dump(dataset_dict, f)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
