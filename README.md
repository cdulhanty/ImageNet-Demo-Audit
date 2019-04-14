# ImageNet-Demo-Audit

### Data Download

[Kaggle is the new home of the ImageNet Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview).

Create a Kaggle account and download the ILSVRC images (155 GB) [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/download/imagenet_object_localization.tar.gz); extract images via:

 ~~~~
 tar -xzvf imagenet_object_localization.tar.gz
 ~~~~

### Package Installation

Install requirements via:

~~~~
pip install -r requirements.txt
~~~~

### Face Detection via [FaceBoxes](https://arxiv.org/abs/1708.05234) 
~~~~
git clone https://github.com/zisianw/FaceBoxes.PyTorch.git
rn FaceBoxes.PyTorch FaceBoxes
cd FaceBoxes
./make.sh
~~~~

Download [Face Detection Data Set and Benchmark](http://vis-www.cs.umass.edu/fddb/) dataset [here](http://tamaraberg.com/faceDataset/originalPics.tar.gz); extract images via:
~~~~
mkdir data/FDDB/images/
tar -xzvf imagenet_object_localization.tar.gz
~~~~

### Skin Segmentation via  

~~~~
git clone https://github.com/noelcodella/segmentation-keras-tensorflow.git
rndir segmentation-keras-tensorflow segmentation_keras_tensorflow
~~~~

## Apparent Age and Gender Estimation via [DEX](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

Download the pre-trained age and gender estimation models, convert to PyTorch

Via this link: https://colab.research.google.com/drive/1l4Z7_IjTG7Z1KpmhyWFEWlozxM9CvJn_

Download [age.py, age.npy. gender.py, gender.npy]
