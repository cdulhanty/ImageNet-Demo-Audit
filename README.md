# ImageNet-Demo-Audit

- Download from Kaggle (requires an account) the [ILSVRC images here](https://www.kaggle.com/c/imagenet-object-localization-challenge/download/imagenet_object_localization.tar.gz) and the [synset matching file here](https://www.kaggle.com/c/imagenet-object-localization-challenge/download/LOC_synset_mapping.txt); extract images via:

 ~~~~
 tar xvzf imagenet_object_localization.tar.gz
 ~~~~

- Install requirements via:

~~~~
pip install -r requirements.txt
~~~~
    
- Face Detection via [facessd_mobilenet_v2_quantized_open_image_v4](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- Age Estimation via DEX


