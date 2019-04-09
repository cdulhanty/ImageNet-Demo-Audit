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

### Tensorflow Object Detection API Install


#### Protobuf Compilation

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. Before the framework can be used, the Protobuf libraries
must be compiled. This should be done by running the following command from
the models/research/ directory:


``` bash
# From models/research/
protoc object_detection/protos/*.proto --python_out=.
```


#### Add Libraries to PYTHONPATH

When running locally, the models/research/ and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
models/research/:


``` bash
# From models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file, replacing \`pwd\` with the absolute path of
models/research on your system.

#### Testing the Installation

You can test that you have correctly installed the Tensorflow Object Detection\
API by running the following command:

```bash
python object_detection/builders/model_builder_test.py
```

Might have to run this command if your machine is kinda messed up, like mine is, (i.e. 'ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory'):

```bash
sudo ldconfig /usr/local/cuda/lib64
```

### Unconstrained Face Detection via [facessd_mobilenet_v2_quantized_open_image_v4](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#open-images-trained-models)

Download model from [here](http://download.tensorflow.org/models/object_detection/facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz); extract via:

 ~~~~
 tar -xzvf facessd_mobilenet_v2_quantized_320x320_open_image_v4.tar.gz
 ~~~~
  

- Age Estimation via DEX


