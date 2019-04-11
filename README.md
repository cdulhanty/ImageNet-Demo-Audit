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




- Age Estimation via DEX


