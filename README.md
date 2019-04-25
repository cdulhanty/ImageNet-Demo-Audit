# ImageNet-Demo-Audit

## Data Download

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

Use [Face Detection Data Set and Benchmark (FDDB)](http://vis-www.cs.umass.edu/fddb/) for evaluation.

[Download here](http://tamaraberg.com/faceDataset/originalPics.tar.gz), extract images via:
~~~~
mkdir data/FDDB/images/
tar -xzvf imagenet_object_localization.tar.gz
~~~~

Download FaceBoxes weights from Google Drive, run on FDDB
~~~~
python3 test.py --dataset FDDB
~~~~

Download [FDDB evaluation code](http://vis-www.cs.umass.edu/fddb/evaluation.tgz) ... actually, just clone this repo, b/c there are changes to the Makefile and common.cpp that are important to change for it to work!
~~~~
cd evaluate
make
./evaluate -a ../FaceBoxes/data/FDDB/annotations.txt -d ../FaceBoxes/eval/FDDB_dets.txt -f 0 -i ../FaceBoxes/data/FDDB/images/ -l ../FaceBoxes/data/FDDB/img_list.txt -r [group]
cd ../face-eval/
python2 plot_AP_fddb.py ../evaluation/[group]DiscROC.txt
~~~~

## Apparent Age and Gender Estimation via [DEX](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

Download the pre-trained age and gender estimation models, convert to PyTorch via [Google Colab](https://colab.research.google.com/drive/1l4Z7_IjTG7Z1KpmhyWFEWlozxM9CvJn_)

Download [age.py, age.pth. gender.py, gender.pth]
