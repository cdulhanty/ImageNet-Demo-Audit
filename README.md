# ImageNet-Demo-Audit

## ILSVRC Dataset Download

[Kaggle is the new home of the ImageNet Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview).

Create a Kaggle account and download the ILSVRC images (155 GB) [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/download/imagenet_object_localization.tar.gz); extract images via:

 ~~~~
 tar -xzvf imagenet_object_localization.tar.gz
 ~~~~

## Package Installation

Install requirements via:

~~~~
pip install -r requirements.txt
~~~~

## Face Detection via [FaceBoxes](https://arxiv.org/abs/1708.05234) 
~~~~
git clone https://github.com/zisianw/FaceBoxes.PyTorch.git
rn FaceBoxes.PyTorch FaceBoxes
cd FaceBoxes
./make.sh
~~~~

Download FaceBoxes weights from [Google Drive](https://drive.google.com/open?id=1eyqFViMoBlN8JokGRHxbnJ8D4o0pTWac)


## Apparent Age and Gender Estimation via [DEX](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

Download the pre-trained age and gender estimation models, convert to PyTorch via this [Google Colab Page](https://colab.research.google.com/drive/1l4Z7_IjTG7Z1KpmhyWFEWlozxM9CvJn_)

Download [age.py, age.pth. gender.py, gender.pth]
