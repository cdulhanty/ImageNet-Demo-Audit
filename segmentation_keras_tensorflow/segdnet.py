# Noel C. F. Codella
# Example Semantic Segmentation Code for Keras / TensorFlow

# GLOBAL DEFINES
T_G_WIDTH = 224
T_G_HEIGHT = 224
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

T_G_CHUNKSIZE = 5000

USAGE_LEARN = 'Usage: \n\t ' \
              '-learn <Train Images (TXT)> <Train Masks (TXT)> <Val Images (TXT)> <Val Masks (TXT)> ' \
              '<batch size> <num epochs> <output model prefix> <option: load weights from...> \n\t' \
              '-extract <Model Prefix> <Input Image List (TXT)> <Output File (TXT)> \n\t\t' \
              'Builds and scores a model'

# Misc. Necessities
import sys
import os
import ssl # these two lines solved issues loading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imresize
np.random.seed(T_G_SEED)
import scipy

# TensorFlow Includes
import tensorflow as tf
#from tensorflow.contrib.losses import metric_learning
tf.set_random_seed(T_G_SEED)

# Keras Imports & Defines 
import keras
import keras.applications
import keras.optimizers
import keras.losses
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl

from keras.preprocessing.image import ImageDataGenerator

# Uncomment to use the TensorFlow Debugger
#from tensorflow.python import debug as tf_debug
#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)

# Generator object for data augmentation.
# Can change values here to affect augmentation style.
datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=0.15,
                             height_shift_range=0.15,
                             zoom_range=0.15,
                             horizontal_flip=True,
                             vertical_flip=True,)


# A binary jaccard (non-differentiable)
def jaccard_index_b(y_true, y_pred):

    safety = 0.001

    y_true_f = K.cast(K.greater(K.flatten(y_true),0.5),'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred),0.5),'float32')

    top = K.sum(K.minimum(y_true_f, y_pred_f))
    bottom = K.sum(K.maximum(y_true_f, y_pred_f))

    return top / (bottom + safety)


# A binary jaccard (non-differentiable)
def jaccard_loss_b(y_true, y_pred):

    return 1 - jaccard_index_b(y_true, y_pred)


# An example loss based on multiple metrics
def joint_loss(y_true, y_pred):

    return 0.4 * jaccard_loss_b(y_true, y_pred) + 0.2 * soft_jaccard_loss(y_true, y_pred) + 0.2 * jaccard_loss(y_true, y_pred) + 0.2 * keras.losses.mean_squared_error(y_true, y_pred)


# A computation of the jaccard index
def jaccard_index(y_true, y_pred):
    
    safety = 0.001

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    top = K.sum(K.minimum(y_true_f, y_pred_f))
    bottom = K.sum(K.maximum(y_true_f, y_pred_f))

    return top / (bottom + safety)


# An example loss based on jaccard index
def jaccard_loss(y_true, y_pred):

    return 1 - jaccard_index(y_true, y_pred)


# a 'soft' version of the jaccard index
def soft_jaccard_index(y_true, y_pred):

    safety = 0.001

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    top = K.sum(y_true_f * y_pred_f)
    bottom = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - top

    return top / (bottom + safety)


def soft_jaccard_loss(y_true, y_pred):

    return 1 - soft_jaccard_index(y_true, y_pred)


# generator function for data augmentation
def createDataGen(X, Y, b):

    local_seed = T_G_SEED
    genX = datagen.flow(X,Y, batch_size=b, seed=local_seed, shuffle=False)
    genY = datagen.flow(Y,Y, batch_size=b, seed=local_seed, shuffle=False)
    while True:
            Xi = genX.next()
            Yi = genY.next()

            yield Xi[0], Yi[0]


def createModel():

    # Initialize a Model
    net_input = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
    #net_model = keras.applications.densenet.DenseNet121(weights='imagenet', include_top = False, input_tensor=net_input)
    net_model = keras.applications.densenet.DenseNet201(weights='imagenet', include_top = False, input_tensor=net_input)

    numfilters = 128
    gnoise = 0.001

    # New Layers 
    net = net_model.output

    #up0 = kl.UpSampling2D((2,2))(net)
    up0 = kl.Conv2DTranspose(numfilters, (3, 3), strides=(2, 2), padding='same')(net)
    up0 = kl.BatchNormalization()(up0)
    up0 = kl.GaussianNoise(gnoise)(up0)
    up0 = kl.Activation('elu')(up0)
    skip = kl.Conv2D(numfilters, (1,1), padding='same')(net_model.get_layer("pool4_relu").output)
    skip = kl.BatchNormalization()(skip)
    skip = kl.Activation('elu')(skip)
    up = kl.Concatenate(axis=3)([up0, skip])
    #up = up0
    up = kl.SpatialDropout2D(0.5,data_format='channels_last')(up)
    up = kl.Conv2D(numfilters, (3,3), padding='same', name='up1a')(up)
    up = kl.BatchNormalization()(up)
    up = kl.GaussianNoise(gnoise)(up)
    up = kl.Activation('elu')(up)
    up = kl.Conv2D(numfilters, (3,3), padding='same', name='up1b')(up)
    up = kl.BatchNormalization()(up)
    up = kl.GaussianNoise(gnoise)(up)
    up = kl.Activation('elu')(up)

    #up1 = kl.UpSampling2D((2,2))(up)
    up1 = kl.Conv2DTranspose(int(numfilters/2.0), (3, 3), strides=(2, 2), padding='same')(up)
    up1 = kl.BatchNormalization()(up1)
    up1 = kl.GaussianNoise(gnoise)(up1)
    up1 = kl.Activation('elu')(up1)
    skip = kl.Conv2D(int(numfilters/2.0), (1,1), padding='same')(net_model.get_layer("pool3_relu").output)
    skip = kl.BatchNormalization()(skip)
    skip = kl.Activation('elu')(skip)
    up = kl.Concatenate(axis=3)([up1, skip])
    #up = up1
    #up = kl.SpatialDropout2D(0.5,data_format='channels_last')(up)
    up = kl.Conv2D(int(numfilters/2.0), (3,3), padding='same', name='up2a')(up)
    up = kl.BatchNormalization()(up)
    up = kl.GaussianNoise(gnoise)(up)
    up = kl.Activation('elu')(up)
    up = kl.Conv2D(int(numfilters/2.0), (3,3), padding='same', name='up2b')(up)
    up = kl.BatchNormalization()(up)
    up = kl.GaussianNoise(gnoise)(up)
    up = kl.Activation('elu')(up)

    #up2 = kl.UpSampling2D((2,2))(up)
    up2 = kl.Conv2DTranspose(int(numfilters/4.0), (3, 3), strides=(2, 2), padding='same')(up)
    up2 = kl.BatchNormalization()(up2)
    up2 = kl.GaussianNoise(gnoise)(up2)
    up2 = kl.Activation('elu')(up2)
    skip = kl.Conv2D(int(numfilters/4.0), (1,1), padding='same')(net_model.get_layer("pool2_relu").output)
    skip = kl.BatchNormalization()(skip)
    skip = kl.Activation('elu')(skip)
    up = kl.Concatenate(axis=3)([up2, skip])
    #up = kl.SpatialDropout2D(0.5,data_format='channels_last')(up)
    #up = up2
    up = kl.Conv2D(int(numfilters/4.0), (3,3), padding='same', name='up3a')(up)
    up = kl.BatchNormalization()(up)
    up = kl.GaussianNoise(gnoise)(up)
    up = kl.Activation('elu')(up)
    up = kl.Conv2D(int(numfilters/4.0), (3,3), padding='same', name='up3b')(up)
    up = kl.BatchNormalization()(up)
    up = kl.GaussianNoise(gnoise)(up)
    up = kl.Activation('elu')(up)

    #side = kl.Conv2D(numfilters/8, (1,1), padding='same', name='side1')(net_model.get_layer("conv1/relu").output)
    #side = kl.BatchNormalization()(side)
    #side = kl.GaussianNoise(gnoise)(side)
    #side = kl.Activation('elu')(side)
    #side = kl.Conv2D(numfilters/8, (3,3), padding='same', name='side2')(side)
    #side = kl.BatchNormalization()(side)
    #side = kl.GaussianNoise(gnoise)(side)
    #side = kl.Activation('elu')(side) 

    #up3 = kl.UpSampling2D((2,2))(up)
    up3 = kl.Conv2DTranspose(int(numfilters/8.0), (3, 3), strides=(2, 2), padding='same')(up)
    up3 = kl.BatchNormalization()(up3)
    up3 = kl.GaussianNoise(gnoise)(up3)
    up3 = kl.Activation('elu')(up3)
    #up = kl.Concatenate(axis=3)([up3, side])
    #up = kl.SpatialDropout2D(0.5,data_format='channels_last')(up3)
    up = up3
    up = kl.Conv2D(int(numfilters/8.0), (3,3), padding='same', name='up4a')(up)
    up = kl.BatchNormalization()(up)  
    up = kl.GaussianNoise(gnoise)(up) 
    up = kl.Activation('elu')(up)
    up = kl.Conv2D(int(numfilters/8.0), (3,3), padding='same', name='up4b')(up)
    up = kl.BatchNormalization()(up)
    up = kl.GaussianNoise(gnoise)(up)
    up = kl.Activation('elu')(up)

    #side = kl.Conv2D(numfilters/8, (3,3), padding='same', name='side1')(net_model.get_layer("input_1").output)
    #side = kl.BatchNormalization()(side)
    #side = kl.Activation('elu')(side)
    #side = kl.Conv2D(numfilters/8, (3,3), padding='same', name='side2')(side)
    #side = kl.BatchNormalization()(side)
    #side = kl.Activation('elu')(side)

    #up4 = kl.UpSampling2D((2,2))(up)
    up4 = kl.Conv2DTranspose(int(numfilters/16.0), (3, 3), strides=(2, 2), padding='same')(up)
    up4 = kl.BatchNormalization()(up4)
    up4 = kl.GaussianNoise(gnoise)(up4)
    up4 = kl.Activation('elu')(up4)
    #up = kl.Concatenate(axis=3)([up4, side, kl.UpSampling2D((16,16))(up0), kl.UpSampling2D((8,8))(up1), kl.UpSampling2D((4,4))(up2),kl.UpSampling2D((2,2))(up3)])
    #up = kl.SpatialDropout2D(0.5,data_format='channels_last')(up4)
    up = up4
    up = kl.Conv2D(int(numfilters/16.0), (3,3), padding='same', name='up5a')(up)
    up = kl.BatchNormalization()(up)
    up = kl.Activation('elu')(up)
    up = kl.Conv2D(int(numfilters/16.0), (3,3), padding='same', name='up5b')(up)
    bn0 = kl.BatchNormalization()(up)
    top = kl.Activation('elu')(bn0)
    #up = kl.SpatialDropout2D(0.5,data_format='channels_last')(up)

    outnet = kl.Conv2D(1, (1,1), padding='same', name='segout')(top)
    #outnet = kl.BatchNormalization()(outnet)
    outnet = kl.Activation('sigmoid')(outnet)

    # model creation
    base_model = Model(net_model.input, outnet, name="base_model")

    print(base_model.summary())

    base_model.compile(optimizer=keras.optimizers.Adadelta(decay=0.0), loss=jaccard_loss, metrics=[jaccard_loss_b, keras.losses.binary_crossentropy, joint_loss, keras.losses.mean_squared_error, soft_jaccard_loss, jaccard_loss])

    return base_model


def t_save_image_list(inputimagelist, start, length, pred, outputpath, rdim=T_G_WIDTH):

    # Count the number of images in the list
    list_file = open(inputimagelist, "r")
    content = list_file.readlines()
    content = content[start:start+length]

    c_img = 0
    for img_file in content:
        img_file = img_file.rstrip('\n')
        filename = outputpath + "/" + img_file
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        outimg = (pred[c_img,:,:,:])*255.
        cv2.imwrite(filename, outimg)
        c_img = c_img + 1


# loads an image and preprocesses
def t_read_image(loc):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (T_G_HEIGHT,T_G_WIDTH))
    t_image = t_image.astype("float32")
    t_image = keras.applications.densenet.preprocess_input(t_image, data_format='channels_last')

    return t_image


def t_norm_image(img):
    new_img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    return new_img


# loads a set of images from a text index file   
def t_read_image_list(flist, start, length, color=1, norm=0):

    with open(flist) as f:
        content = f.readlines() 
    content = [x.strip().split()[0] for x in content] 

    datalen = length
    if (datalen < 0):
        datalen = len(content)

    if (start + datalen > len(content)):
        datalen = len(content) - start

    if (color == 1):
        imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, T_G_NUMCHANNELS))
    else:
        imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, 1))

    for i in range(start, start+datalen):
        if ((i-start) < len(content)):
            val = t_read_image(content[i])
            if (color == 0):
                val = val[:,:,0]
                val = np.expand_dims(val,2)
            imgset[i-start] = val
            if (norm == 1):
                imgset[i-start] = (t_norm_image(imgset[i-start]) * 1.0 + 0.0) 

    return imgset


def file_numlines(fn):
    with open(fn) as f:
        return sum(1 for _ in f)


def main(argv):

    if len(argv) < 2:
        print(USAGE_LEARN)
        return

    if 'learn' in argv[0]:
        learn(argv[1:])
    elif 'extract' in argv[0]:
        extract(argv[1:])    

    return


def extract(argv):

    if len(argv) < 3:
        print('Usage: \n\t <Model Prefix> <Input Image List (TXT)> <Output Path> \n\t\tExtracts model')
        return

    modelpref = argv[0]
    imglist = argv[1]
    outfile = argv[2]

    with open(modelpref + '.json', "r") as json_file:
        model_json = json_file.read()

    loaded_model = keras.models.model_from_json(model_json)
    loaded_model.load_weights(modelpref + '.h5')

    base_model = loaded_model 

    scoreModel(imglist,base_model,outfile)

    return


def scoreModel(imglist, base_model, outfile, aug=0):

    chunksize = T_G_CHUNKSIZE
    total_img = file_numlines(imglist)
    total_img_ch = int(np.ceil(total_img / float(chunksize)))

    for i in range(0, total_img_ch):
        imgs = t_read_image_list(imglist, i*chunksize, chunksize)
        valsa = base_model.predict(imgs)
        
        # test time data augmentation
        if (aug > 0):
            valsb = base_model.predict(scipy.ndimage.rotate(imgs, 90, axes=(2,1), reshape=False))
            valsb = scipy.ndimage.rotate(valsb, 270, axes=(2,1), reshape=False)
            valsc = base_model.predict(scipy.ndimage.rotate(imgs, 180, axes=(2,1), reshape=False))
            valsc = scipy.ndimage.rotate(valsc, 180, axes=(2,1), reshape=False)
            valsd = base_model.predict(scipy.ndimage.rotate(imgs, 270, axes=(2,1), reshape=False))
            valsd = scipy.ndimage.rotate(valsd, 90, axes=(2,1), reshape=False)
            #valse = base_model.predict(np.roll(imgs, 10, axis=2))
            #valse = np.roll(valse, -10, axis=2)
            #valsf = base_model.predict(np.roll(imgs, 10, axis=1))
            #valsf = np.roll(valsf, -10, axis=1)

            vals = (valsa + valsb + valsc + valsd) / 4.0

        else:
            vals = valsa

        t_save_image_list(imglist, i*chunksize, chunksize, vals, outfile)

    return


def learn(argv):
    
    if len(argv) < 6:
        print(USAGE_LEARN)
        return

    in_t_i = argv[0]
    in_t_m = argv[1]

    in_v_i = argv[2]
    in_v_m = argv[3]

    batch = int(argv[4])
    numepochs = int(argv[5])
    outpath = argv[6] 

    # chunksize is the number of images we load from disk at a time
    chunksize = T_G_CHUNKSIZE
    total_t = file_numlines(in_t_i)
    total_v = file_numlines(in_v_i)
    total_t_ch = int(np.ceil(total_t / float(chunksize)))
    total_v_ch = int(np.ceil(total_v / float(chunksize)))

    print('Dataset has ' + str(total_t) + ' training, and ' + str(total_v) + ' validation.')

    print( 'Creating a model ...')
    model = createModel()

    if len(argv) > 7:
        print('Loading weights from: ' + argv[7] + ' ... ')
        model.load_weights(argv[7])

    print('Training loop ...')
   
    images_t = []
    masks_t = []
    images_v = []
    masks_v = []

    t_imloaded = 0
    v_imloaded = 0
 
    # manual loop over epochs to support very large sets 
    for e in range(0, numepochs):

        for t in range(0, total_t_ch):

            print('Epoch ' + str(e) + ': train chunk ' + str(t+1) + '/ ' + str(total_t_ch) + ' ...')

            if ( t_imloaded == 0 or total_t_ch > 1 ): 
                print('Reading image lists ...')
                images_t = t_read_image_list(in_t_i, t*chunksize, chunksize)
                masks_t = t_read_image_list(in_t_m, t*chunksize, chunksize, 0, 1)
                t_imloaded = 1

            print('Starting to fit ...')

            # This method uses data augmentation
            model.fit_generator(generator=createDataGen(images_t,masks_t,batch), steps_per_epoch=len(images_t) / batch, epochs=1, shuffle=False, use_multiprocessing=True)
        
        # In case the validation images don't fit in memory, we load chunks from disk again. 
        val_res = [0.0, 0.0]
        total_w = 0.0
        for v in range(0, total_v_ch):

            print('Epoch ' + str(e) + ': val chunk ' + str(v+1) + '/ ' + str(total_v_ch) + ' ...')

            if ( v_imloaded == 0 or total_v_ch > 1 ):
                print('Loading validation image lists ...')
                images_v = t_read_image_list(in_v_i, v*chunksize, chunksize)
                masks_v = t_read_image_list(in_v_m, v*chunksize, chunksize, 0, 1)
                v_imloaded = 1

            # Weight of current validation measurement. 
            # if loaded expected number of items, this will be 1.0, otherwise < 1.0, and > 0.0.
            w = float(images_v.shape[0]) / float(chunksize)
            total_w = total_w + w

            curval = model.evaluate(images_v, masks_v, batch_size=batch)
            val_res[0] = val_res[0] + w*curval[0]
            val_res[1] = val_res[1] + w*curval[1]

        val_res = [x / total_w for x in val_res]

        print('Validation Results: ' + str(val_res))

    print('Saving model ...')

    # Save the model and weights
    model.save(outpath + '.h5')

    # Due to some remaining Keras bugs around loading custom optimizers
    # and objectives, we save the model architecture as well
    model_json = model.to_json()
    with open(outpath + '.json', "w") as json_file:
        json_file.write(model_json)

    # scoreModel(in_v_i, model, 'debugout')

    return


# Main Driver
if __name__ == "__main__":
    main(sys.argv[1:])
