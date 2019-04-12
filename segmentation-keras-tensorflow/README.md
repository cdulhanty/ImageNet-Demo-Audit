# segmentation-keras-tensorflow
An example of semantic segmentation using a fully convolutional framework in keras-tensorflow. 

Network is fine-tuned from DenseNet, initialized from ImageNet pretraining. Decoder uses skip-connections, similar to U-Net, from the encoder. Example includes several example objective functions one can experiment with, and test time augmentation (rotations). 

Two versions are presented: one for binary class segmentation, the other for multiclass (mc).


**Example Models:**

- Human Skin Segmentation (for segdnet.py): https://ibm.box.com/s/r0irrizstbqahfkwmf40hh7j1wxr4fxm
