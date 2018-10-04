from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import h5py

from keras.models import Model
from keras.models import load_model
from keras.layers.core import Dropout, Reshape
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate,Activation
import utils.template_match_target as tmt
import utils.processing as proc


# Check Keras version - code will switch API if needed.
from keras import __version__ as keras_version
k2 = True if keras_version[0] == '2' else False

# If Keras is v2.x.x, create Keras 1-syntax wrappers.
if not k2:
    from keras.layers import merge, Input
    from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                            UpSampling2D)

else:
    from keras.layers import Concatenate, Input
    from keras.layers.convolutional import (Conv2D, MaxPooling2D,
                                            UpSampling2D)

    def merge(layers, mode=None, concat_axis=None):
        """Wrapper for Keras 2's Concatenate class (`mode` is discarded)."""
        return Concatenate(axis=concat_axis)(list(layers))

    def Convolution2D(n_filters, FL, FLredundant, activation=None,
                      init=None, W_regularizer=None, border_mode=None):
        """Wrapper for Keras 2's Conv2D class."""
        return Conv2D(n_filters, FL, activation=activation,
                      kernel_initializer=init,
                      kernel_regularizer=W_regularizer,
                      padding=border_mode)

########################################################################################


class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0:
            name = 'weights%08d.h5' % self.batch
            self.model.save(name)
        self.batch += 1
        
########################################################################################
########################
def get_param_i(param, i):
    """Gets correct parameter for iteration i.

    Parameters
    ----------
    param : list
        List of model hyperparameters to be iterated over.
    i : integer
        Hyperparameter iteration.

    Returns
    -------
    Correct hyperparameter for iteration i.
    """
    if len(param) > i:
        return param[i]
    else:
        return param[0]

########################
def custom_image_generator(data, target, batch_size=32):
    """Custom image generator that manipulates image/target pairs to prevent
    overfitting in the Convolutional Neural Network.

    Parameters
    ----------
    data : array
        Input images.
    target : array
        Target images.
    batch_size : int, optional
        Batch size for image manipulation.

    Yields
    ------
    Manipulated images and targets.
        
    """
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i + batch_size].copy(), target[i:i + batch_size].copy()

            # Random color inversion
            # for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
            #     d[j][d[j] > 0.] = 1. - d[j][d[j] > 0.]

            # Horizontal/vertical flips
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])      # left/right
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])      # up/down

            # Random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix, npix + 1, batch_size)    # Horizontal shift
            v = np.random.randint(-npix, npix + 1, batch_size)    # Vertical shift
            r = np.random.randint(0, 4, batch_size)               # 90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
                              mode='constant')[npix + h[j]:L + h[j] + npix,
                                               npix + v[j]:W + v[j] + npix, :]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix,
                                                              npix + v[j]:W + v[j] + npix]
                d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
            yield (d, t)

########################
def get_metrics(data, craters, dim, model, beta=1):
    """Function that prints pertinent metrics at the end of each epoch.

    Parameters
    ----------
    data : hdf5
        Input images.
    craters : hdf5
        Pandas arrays of human-counted crater data.
    dim : int
        Dimension of input images (assumes square).
    model : keras model object
        Keras model
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    X, Y = data[0], data[1]

    # Get csvs of human-counted craters
    csvs = []
    minrad, maxrad, cutrad, n_csvs = 1, 50, 0.8, len(X)
    diam = 'Diameter (pix)'
    
    
    
    for i in range(n_csvs):
        csv = craters[proc.get_id(i,5)]
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
    print("")
    print("*********Custom Loss*********")
    recall, precision, fscore = [], [], []
    frac_new, frac_new2, maxrad = [], [], []
    err_lo, err_la, err_r = [], [], []
    frac_duplicates = []
    preds = model.predict(X)
    for i in range(n_csvs):
        if len(csvs[i]) < 3:
            continue
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, frac_dupes) = tmt.template_match_t2c(preds[i], csvs[i],
                                                            rmv_oor_csvs=0)
        if N_match > 0:
            p = float(N_match) / float(N_match + (N_detect - N_match))
            r = float(N_match) / float(N_csv)
            f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
            diff = float(N_detect - N_match)
            fn = diff / (float(N_detect) + diff)
            fn2 = diff / (float(N_csv) + diff)
            recall.append(r)
            precision.append(p)
            fscore.append(f)
            frac_new.append(fn)
            frac_new2.append(fn2)
            maxrad.append(maxr)
            err_lo.append(elo)
            err_la.append(ela)
            err_r.append(er)
            frac_duplicates.append(frac_dupes)
        else:
            print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
                  (i, N_csv, N_detect, N_match))

    print("binary XE score = %f" % model.evaluate(X, Y))
    if len(recall) > 3:
        print("mean and std of N_match/N_csv (recall) = %f, %f" %
              (np.mean(recall), np.std(recall)))
        print("""mean and std of N_match/(N_match + (N_detect-N_match))
              (precision) = %f, %f""" % (np.mean(precision), np.std(precision)))
        print("mean and std of F_%d score = %f, %f" %
              (beta, np.mean(fscore), np.std(fscore)))
        print("""mean and std of (N_detect - N_match)/N_detect (fraction
              of craters that are new) = %f, %f""" %
              (np.mean(frac_new), np.std(frac_new)))
        print("""mean and std of (N_detect - N_match)/N_csv (fraction of
              "craters that are new, 2) = %f, %f""" %
              (np.mean(frac_new2), np.std(frac_new2)))
        print("median and IQR fractional longitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_lo), np.percentile(err_lo, 25),
               np.percentile(err_lo, 75)))
        print("median and IQR fractional latitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_la), np.percentile(err_la, 25),
               np.percentile(err_la, 75)))
        print("median and IQR fractional radius diff = %f, 25:%f, 75:%f" %
              (np.median(err_r), np.percentile(err_r, 25),
               np.percentile(err_r, 75)))
        print("mean and std of frac_duplicates: %f, %f" %
              (np.mean(frac_duplicates), np.std(frac_duplicates)))
        print("""mean and std of maximum detected pixel radius in an image =
              %f, %f""" % (np.mean(maxrad), np.std(maxrad)))
        print("""absolute maximum detected pixel radius over all images =
              %f""" % np.max(maxrad))
        print("")

########################
def build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters):
    """Function that builds the (UNET) convolutional neural network.

    Parameters
    ----------
    dim : int
        Dimension of input images (assumes square).
    learn_rate : float
        Learning rate.
    lmbda : float
        Convolution2D regularization parameter.
    drop : float
        Dropout fraction.
    FL : int
        Filter length.
    init : string
        Weight initialization type.
    n_filters : int
        Number of filters in each layer.

    Returns
    -------
        
    model : keras model object
        Constructed Keras model.
    """
    print('Making UNET model...')
    img_input = Input(batch_shape=(None, dim, dim, 1))

#    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
#                       W_regularizer=l2(lmbda), border_mode='same')(img_input)
#    a1 = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
#                       W_regularizer=l2(lmbda), border_mode='same')(a1)
#    a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)
#
#    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
#                       W_regularizer=l2(lmbda), border_mode='same')(a1P)
#    a2 = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
#                       W_regularizer=l2(lmbda), border_mode='same')(a2)
#    a2P = MaxPooling2D((2, 2), strides=(2, 2))(a2)
#
#    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
#                       W_regularizer=l2(lmbda), border_mode='same')(a2P)
#    a3 = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
#                       W_regularizer=l2(lmbda), border_mode='same')(a3)
#    a3P = MaxPooling2D((2, 2), strides=(2, 2),)(a3)
#
#    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(a3P)
#    u = Convolution2D(n_filters * 4, FL, FL, activation='relu', init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(u)
#
#    u = UpSampling2D((2, 2))(u)
#    u = merge((a3, u), mode='concat', concat_axis=3)
#    u = Dropout(drop)(u)
#    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(u)
#    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(u)
#
#    u = UpSampling2D((2, 2))(u)
#    u = merge((a2, u), mode='concat', concat_axis=3)
#    u = Dropout(drop)(u)
#    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(u)
#    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(u)
#
#    u = UpSampling2D((2, 2))(u)
#    u = merge((a1, u), mode='concat', concat_axis=3)
#    u = Dropout(drop)(u)
#    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(u)
#    u = Convolution2D(n_filters, FL, FL, activation='relu', init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(u)
#
#    # Final output
#    final_activation = 'sigmoid'
#    u = Convolution2D(1, 1, 1, activation=final_activation, init=init,
#                      W_regularizer=l2(lmbda), border_mode='same')(u)
#    u = Reshape((dim, dim))(u)
#    if k2:
#        model = Model(inputs=img_input, outputs=u)
#    else:
#        model = Model(input=img_input, output=u)
#
#    optimizer = Adam(lr=learn_rate)
#    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    
#    22222222----------------------------------------------------------------------------------------2222222222222222222222222222222
    # 1024
#    down0 = Convolution2D(64,3,3,activation = 'relu',init=init,W_regularizer=l2(lmbda),border_mode='same')(img_input)
##    down0 = Convolution2D(64,3,3,activation = 'relu',init=init,W_regularizer=l2(lmbda),border_mode='same')(img_input)
##    plus_0 = Convolution2D(32,2,2,border_mode='same')(down0)
##a1P = MaxPooling2D((2, 2), strides=(2, 2))(a1)
#    down0 = MaxPooling2D((2, 2), strides=(2, 2))(down0)
#    down0 = Activation('relu')(down0)
#
#    # 512
#
#
#    down1 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(down0)
#    down1 = Convolution2D(32,2,2,border_mode='same')(down1)
#    plus_1 = merge((down1, down0), mode='concat', concat_axis=3)
#    down1= MaxPooling2D((2, 2), strides=(2, 2))(plus_1)
#    down1 = Activation('relu')(down1)
#
#    # 256
#
#    down2 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(down1)
#    down2 = Convolution2D(32,2,2,border_mode='same')(down2)
#    plus_2 = merge((down2, down1), mode='concat', concat_axis=3)
#    down2= MaxPooling2D((2, 2), strides=(2, 2))(plus_2)
#    down2 = Activation('relu')(down2)
#
#    # 128
#    
#    down3 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(down2)
#    down3 = Convolution2D(32,2,2,border_mode='same')(down3)
#    plus_3 = merge((down3, down2), mode='concat', concat_axis=3)
#    down3= MaxPooling2D((2, 2), strides=(2, 2))(plus_3)
#    down3 = Activation('relu')(down3)
#
#    # 64
#
#    
#    down4 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(down3)
#    down4 = Convolution2D(32,2,2,border_mode='same')(down4)
#    plus_4 = merge((down4, down3), mode='concat', concat_axis=3)
#    down4= MaxPooling2D((2, 2), strides=(2, 2))(plus_4)
#    down4 = Activation('relu')(down4)
#
#    # 32
#
#
#    down5 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(down4)
#    down5 = Convolution2D(32,2,2,border_mode='same')(down5)
#    plus_5 = merge((down5, down4), mode='concat', concat_axis=3)
#    down5= MaxPooling2D((2, 2), strides=(2, 2))(plus_5)
#    down5 = Activation('relu')(down5)
#
#    # 16
#
#
#    down6 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(down5)
#    down6 = Convolution2D(32,2,2,border_mode='same')(down6)
#    plus_6 = merge((down6, down5), mode='concat', concat_axis=3)
#    down6= MaxPooling2D((2, 2), strides=(2, 2))(plus_6)
#    down6 = Activation('relu')(down6)
#
#    # 8
## u = UpSampling2D((2, 2))(u)
##    u = merge((a3, u), mode='concat', concat_axis=3)
##    u = Dropout(drop)(u)
##    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
##                      W_regularizer=l2(lmbda), border_mode='same')(u)
##    u = Convolution2D(n_filters * 2, FL, FL, activation='relu', init=init,
##                      W_regularizer=l2(lmbda), border_mode='same')(u)
#    up7 = UpSampling2D((2, 2))(down6)
#
#    # 16
#
#
#    up6 = merge((up7,plus_6),mode='concat',concat_axis=3)
#    up6 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(up6)
#    up6 = Convolution2D(32,3,3,border_mode='same')(up6)
#    up6 = merge((up6,up7),mode='concat',concat_axis=3)
#    up6 = Activation('relu')(up6)
#    up6 = UpSampling2D((2, 2))(up6)
#
#    # 32
#
#    
#    up5 = merge((up6,plus_5),mode='concat',concat_axis=3)
#    up5 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(up5)
#    up5 = Convolution2D(32,3,3,border_mode='same')(up5)
#    up5 = merge((up5,up6),mode='concat',concat_axis=3)
#    up5 = Activation('relu')(up5)
#    up5 = UpSampling2D((2, 2))(up5)
#
#    # 64
#
#
#    
#    up4 = merge((up5,plus_4),mode='concat',concat_axis=3)
#    up4 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(up4)
#    up4 = Convolution2D(32,3,3,border_mode='same')(up4)
#    up4 = merge((up4,up5),mode='concat',concat_axis=3)
#    up4 = Activation('relu')(up4)
#    up4 = UpSampling2D((2, 2))(up4)
#
#    # 128
#
#    up3 = merge((up4,plus_3),mode='concat',concat_axis=3)
#    up3 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(up3)
#    up3 = Convolution2D(32,3,3,border_mode='same')(up3)
#    up3 = merge((up3,up4),mode='concat',concat_axis=3)
#    up3 = Activation('relu')(up3)
#    up3 = UpSampling2D((2, 2))(up3)
#    # 256
#
#    
#    up2 = merge((up3,plus_2),mode='concat',concat_axis=3)
#    up2 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(up2)
#    up2 = Convolution2D(32,3,3,border_mode='same')(up2)
#    up2 = merge((up2,up3),mode='concat',concat_axis=3)
#    up2 = Activation('relu')(up2)
#    up2 = UpSampling2D((2, 2))(up2)
#
#    # 512
#
#    
#    up1 = merge((up2,plus_1),mode='concat',concat_axis=3)
#    up1 = Convolution2D(64,3,3,activation = 'relu',border_mode='same')(up1)
#    up1 = Convolution2D(32,3,3,border_mode='same')(up1)
#    up1 = merge((up1,up2),mode='concat',concat_axis=3)
#    up1 = Activation('relu')(up1)
#    up1 = UpSampling2D((2, 2))(up1)
#
#    # 1024
#
##    classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)
#    classify = Convolution2D(1,1,1,activation = 'sigmoid')(up1)
#    down0 = Conv2D(64, (3, 3), padding='same')(img_input)
#    down0 = Activation('relu')(down0)
#    down0 = Conv2D(64, (3, 3), padding='same')(img_input)
#    down0 = Activation('relu')(down0)
#    plus_0 = Conv2D(32, (2, 2), padding='same')(down0)
#    down0 = MaxPooling2D((2, 2), strides=(2, 2))(down0)
#    down0 = Activation('relu')(down0)
#
#    # 512
#
#    down1 = Conv2D(64, (3, 3), padding='same')(down0)
#    down1 = Activation('relu')(down1)
#    down1 = Conv2D(32, (3, 3), padding='same')(down1)
#    plus_1 = concatenate([down1, down0], axis=3)
#    down1= MaxPooling2D((2, 2), strides=(2, 2))(plus_1)
#    down1 = Activation('relu')(down1)
#
#    # 256
#
#    down2 = Conv2D(64, (3, 3), padding='same')(down1)
#    down2 = Activation('relu')(down2)
#    down2 = Conv2D(32, (2, 2), padding='same')(down2)
#    plus_2 = concatenate([down2, down1], axis=3)
#    down2 = MaxPooling2D((2, 2), strides=(2, 2))(plus_2)
#    down2 = Activation('relu')(down2)
#
#    # 128
#
#    down3 = Conv2D(64, (3, 3), padding='same')(down2)
#    down3 = Activation('relu')(down3)
#    down3 = Conv2D(32, (2, 2), padding='same')(down3)
#    plus_3 = concatenate([down3, down2], axis=3)
#    down3 = MaxPooling2D((2, 2), strides=(2, 2))(plus_3)
#    down3 = Activation('relu')(down3)
#
#    # 64
#
#    down4 = Conv2D(64, (3, 3), padding='same')(down3)
#    down4 = Activation('relu')(down4)
#    down4 = Conv2D(32, (2, 2), padding='same')(down4)
#    plus_4 = concatenate([down4, down3], axis=3)
#    down4 = MaxPooling2D((2, 2), strides=(2, 2))(plus_4)
#    down4 = Activation('relu')(down4)
#
#    # 32
#
#    down5 = Conv2D(64, (3, 3), padding='same')(down4)
#    down5 = Activation('relu')(down5)
#    down5 = Conv2D(32, (2, 2), padding='same')(down5)
#    plus_5 = concatenate([down5, down4], axis=3)
#    down5 = MaxPooling2D((2, 2), strides=(2, 2))(plus_5)
#    down5 = Activation('relu')(down5)
#
#    # 16
#
#    down6 = Conv2D(64, (3, 3), padding='same')(down5)
#    down6 = Activation('relu')(down6)
#    down6 = Conv2D(32, (2, 2), padding='same')(down6)
#    plus_6 = concatenate([down6, down5], axis=3)
#    down6 = MaxPooling2D((2, 2), strides=(2, 2))(plus_6)
#    down6 = Activation('relu')(down6)
#
#    # 8
#
#    up7 = UpSampling2D((2, 2))(down6)
#
#    # 16
#
#    up6 = concatenate([up7, plus_6], axis=3)
#    up6 = Conv2D(64, (3, 3), padding='same')(up6)
#    up6 = Activation('relu')(up6)
#    up6 = Conv2D(32, (3, 3), padding='same')(up6)
#    up6 = concatenate([up6, up7], axis=3)
#    up6 = Activation('relu')(up6)
#    up6 = UpSampling2D((2, 2))(up6)
#
#    # 32
#
#    up5 = concatenate([up6, plus_5], axis=3)
#    up5 = Conv2D(64, (3, 3), padding='same')(up5)
#    up5 = Activation('relu')(up5)
#    up5 = Conv2D(32, (3, 3), padding='same')(up5)
#    up5 = concatenate([up5, up6], axis=3)
#    up5 = Activation('relu')(up5)
#    up5 = UpSampling2D((2, 2))(up5)
#
#    # 64
#
#    up4 = concatenate([up5, plus_4], axis=3)
#    up4 = Conv2D(64, (3, 3), padding='same')(up4)
#    up4 = Activation('relu')(up4)
#    up4 = Conv2D(32, (3, 3), padding='same')(up4)
#    up4 = concatenate([up4, up5], axis=3)
#    up4 = Activation('relu')(up4)
#    up4 = UpSampling2D((2, 2))(up4)
#
#    # 128
#
#    up3 = concatenate([up4, plus_3], axis=3)
#    up3 = Conv2D(64, (3, 3), padding='same')(up3)
#    up3 = Activation('relu')(up3)
#    up3 = Conv2D(32, (3, 3), padding='same')(up3)
#    up3 = concatenate([up3, up4], axis=3)
#    up3 = Activation('relu')(up3)
#    up3 = UpSampling2D((2, 2))(up3)
#
#    # 256
#
#    up2 = concatenate([up3, plus_2], axis=3)
#    up2 = Conv2D(64, (3, 3), padding='same')(up2)
#    up2 = Activation('relu')(up2)
#    up2 = Conv2D(32, (3, 3), padding='same')(up2)
#    up2 = concatenate([up2, up3], axis=3)
#    up2 = Activation('relu')(up2)
#    up2 = UpSampling2D((2, 2))(up2)
#
#    # 512
#
#    up1 = concatenate([up2, plus_1], axis=3)
#    up1 = Conv2D(64, (3, 3), padding='same')(up1)
#    up1 = Activation('relu')(up1)
#    up1 = Conv2D(32, (3, 3), padding='same')(up1)
#    up1 = concatenate([up1, up2], axis=3)
#    up1 = Activation('relu')(up1)
#    up1 = UpSampling2D((2, 2))(up1)
#
#    
#    classify = Conv2D(1, (1, 1), activation='sigmoid')(up1)
#    classify = Reshape((dim, dim))(classify)
#
#
#    model = Model(inputs=img_input, outputs=classify)
#
#    optimizer = Adam(lr=learn_rate)
#    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    droprate=0.25
    n_filters = 32
    upconv = False
    growth_factor = 2
    #inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(img_input)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(droprate)(pool1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(droprate)(pool2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(droprate)(pool3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
    pool4_0 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
    pool4_0 = Dropout(droprate)(pool4_0)

    n_filters *= growth_factor
    pool4_0 = BatchNormalization()(pool4_0)
    conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_0)
    conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
    pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
    pool4_1 = Dropout(droprate)(pool4_1)

    n_filters *= growth_factor
    pool4_1 = BatchNormalization()(pool4_1)
    conv4_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
    conv4_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_2)
    pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_2)
    pool4_2 = Dropout(droprate)(pool4_2)

    n_filters *= growth_factor
    pool4_2 = BatchNormalization()(pool4_2)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Dropout(droprate)(conv5)

    n_filters //= growth_factor
    if upconv:
        up6 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_2])
    else:
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_2])
    up6 = BatchNormalization()(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Dropout(droprate)(conv6)

    n_filters //= growth_factor
    if upconv:
        up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv4_1])
    else:
        up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv4_1])
    up6_1 = BatchNormalization()(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
    conv6_1 = Dropout(droprate)(conv6_1)

    n_filters //= growth_factor
    if upconv:
        up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
    else:
        up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
    up6_2 = BatchNormalization()(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
    conv6_2 = Dropout(droprate)(conv6_2)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Dropout(droprate)(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Dropout(droprate)(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    up9 = BatchNormalization()(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    conv10 = Reshape((256,256))(conv10)

    model = Model(inputs=img_input, outputs=conv10)



    model.compile(optimizer=Adam(), loss='binary_crossentropy')  
    
    print(model.summary())

    return model

########################
def train_and_test_model(Data, Craters, MP, i_MP):
    """Function that trains, tests and saves the model, printing out metrics
    after each model.

    Parameters
    ----------
    Data : dict
        Inputs and Target Moon data.
    Craters : dict
        Human-counted crater data.
    MP : dict
        Contains all relevant parameters.
    i_MP : int
        Iteration number (when iterating over hypers).
    """
    # Static params
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']

    # Iterating params
    FL = get_param_i(MP['filter_length'], i_MP)
    learn_rate = get_param_i(MP['lr'], i_MP)
    n_filters = get_param_i(MP['n_filters'], i_MP)
    init = get_param_i(MP['init'], i_MP)
    lmbda = get_param_i(MP['lambda'], i_MP)
    drop = get_param_i(MP['dropout'], i_MP)

    # Build model
    model = build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters)
# adding a call back to save the model after each 1 epoch ,,
    
    # The path where the the epoch models will be saved
    #edit @Karanraj
    file_path_for_epoch_saving="./epochweights/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
    checkpointsaver =ModelCheckpoint(file_path_for_epoch_saving, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #save best one is kept on.
    #@edit date-29th , adding pretrainded model to improve the accuracy
#    model = load_model('models/model_keras2.h5')
    # Main loop
    n_samples = MP['n_train']
    for nb in range(nb_epoch):
        if k2:
            model.fit_generator(
                custom_image_generator(Data['train'][0], Data['train'][1],
                                       batch_size=bs),
                steps_per_epoch=n_samples/bs, epochs=1, verbose=1,
                # validation_data=(Data['dev'][0],Data['dev'][1]), #no gen
                validation_data=custom_image_generator(Data['dev'][0],
                                                       Data['dev'][1],
                                                       batch_size=bs),
                validation_steps=n_samples,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=3, verbose=0),WeightsSaver(MP['N_save'])])
                    
                    #edit made here @karanraj
        else:
            model.fit_generator(
                custom_image_generator(Data['train'][0], Data['train'][1],
                                       batch_size=bs),
                samples_per_epoch=n_samples, nb_epoch=1, verbose=1,
                # validation_data=(Data['dev'][0],Data['dev'][1]), #no gen
                validation_data=custom_image_generator(Data['dev'][0],
                                                       Data['dev'][1],
                                                       batch_size=bs),
                nb_val_samples=n_samples,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=3, verbose=0),WeightsSaver(MP['N_save'])])

        get_metrics(Data['dev'], Craters['dev'], dim, model)

    if MP['save_models'] == 1:
        model.save(MP['save_dir'])

    print("###################################")
    print("##########END_OF_RUN_INFO##########")
    print("""learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d
          n_train=%d, img_dimensions=%d, init=%s, n_filters=%d, lambda=%e
          dropout=%f""" % (learn_rate, bs, FL, nb_epoch, MP['n_train'],
                           MP['dim'], init, n_filters, lmbda, drop))
    get_metrics(Data['test'], Craters['test'], dim, model)
    print("###################################")
    print("###################################")

########################
def get_models(MP):
    """Top-level function that loads data files and calls train_and_test_model.

    Parameters
    ----------
    MP : dict
        Model Parameters.
    """
    dir = MP['dir']
    n_train, n_dev, n_test = MP['n_train'], MP['n_dev'], MP['n_test']

    # Load data
    train = h5py.File('%strain_images.hdf5' % dir, 'r')
    dev = h5py.File('%sdev_images.hdf5' % dir, 'r')
    test = h5py.File('%stest_images.hdf5' % dir, 'r')
    Data = {
        'train': [train['input_images'][:n_train].astype('float32'),
                  train['target_masks'][:n_train].astype('float32')],
        'dev': [dev['input_images'][:n_dev].astype('float32'),
                  dev['target_masks'][:n_dev].astype('float32')],
        'test': [test['input_images'][:n_test].astype('float32'),
                 test['target_masks'][:n_test].astype('float32')]
    }
    train.close()
    dev.close()
    test.close()

    # Rescale, normalize, add extra dim
    proc.preprocess(Data)

    # Load ground-truth craters
    Craters = {
        'train': pd.HDFStore('%strain_craters.hdf5' % dir, 'r'),
        'dev': pd.HDFStore('%sdev_craters.hdf5' % dir, 'r'),
        'test': pd.HDFStore('%stest_craters.hdf5' % dir, 'r')
    }

    # Iterate over parameters
    for i in range(MP['N_runs']):
        train_and_test_model(Data, Craters, MP, i)


