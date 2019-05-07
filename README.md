# ImageNet-Demo-Audit

## Annotations

Available in Releases Tab: [https://github.com/cdulhanty/ImageNet-Demo-Audit/releases/tag/0.1](https://github.com/cdulhanty/ImageNet-Demo-Audit/releases/tag/0.1)

## Re-implement detections and annotations

### ILSVRC Dataset Download

[Kaggle is the new home of the ImageNet Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview).

Create a Kaggle account and download the ILSVRC images (155 GB) [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/download/imagenet_object_localization.tar.gz); extract images via:

 ~~~~
 tar -xzvf imagenet_object_localization.tar.gz
 ~~~~
 
### ImageNet 'person' Synset Download

ImageNet Fall 2011 Release available at [academic torrents](http://academictorrents.com/details/564a77c1e1119da199ff32622a1609431b9f1c47).

### Package Installation

Install requirements via:

~~~~
pip install -r requirements.txt
~~~~

### Face Detection via [FaceBoxes](https://arxiv.org/abs/1708.05234) 
~~~~
cd FaceBoxes
./make.sh
~~~~

Download FaceBoxes weights from [Google Drive](https://drive.google.com/open?id=1eyqFViMoBlN8JokGRHxbnJ8D4o0pTWac)


### Apparent Age and Gender Estimation via [DEX](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

Download the pre-trained age and gender estimation models available at the above link, specifically:

- Apparent age estimation trained on LAP dataset: **.caffemodel** and **age.prototxt**
- Gender prediction: **.caffemodel** and **gender.prototxt**

Convert to PyTorch models via this [Google Colab Page](https://colab.research.google.com/drive/1l4Z7_IjTG7Z1KpmhyWFEWlozxM9CvJn_)

Download age.py, age.pth. gender.py, gender.pth


### Run Detection & Annotation scripts
Edit file paths in source code for model weights and data locations (TODO: add argparse for command-line input)

~~~~
python face_detection_ImageNet.py
python age_estimation_ImageNet.py
python gender_estimation_ImageNet.py
~~~~
Repeat for person synset.