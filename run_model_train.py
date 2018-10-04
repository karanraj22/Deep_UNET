#!/usr/bin/env python
"""Run Convolutional Neural Network Training

Execute the training of a (UNET) Convolutional Neural Network on
images of the Moon and binary ring targets.
"""

import model3_train as mt

# Model Parameters
MP = {}

# Directory of train/dev/test image and crater hdf5 files.
MP['dir'] = '/home/karan/'

# Image width/height, assuming square images.
MP['dim'] = 256

# Batch size: smaller values = less memory but less accurate gradient estimate
MP['bs'] = 4

# Number of training epochs.
MP['epochs'] = 1

# Number of train/valid/tst samples, needs to be a multiple of batch size.
MP['n_train'] = 1000
MP['n_dev'] = 100
MP['n_test'] = 100

# Save model (binary flag) and directory.
MP['save_models'] = 1
MP['save_dir'] = '/home/karan/deepmars/models/moon3_model.h5'

# Model Parameters (to potentially iterate over, keep in lists).
MP['N_runs'] = 1                # Number of runs
MP['filter_length'] = [3]       # Filter length
MP['lr'] = [0.01]             # Learning rate
MP['n_filters'] = [112]         # Number of filters
MP['init'] = ['he_normal']      # Weight initialization
MP['lambda'] = [1e-6]           # Weight regularization
MP['dropout'] = [0.15]          # Dropout fraction
MP['N_save'] =100 
# Iterating over parameters example.
#    MP['N_runs'] = 2
#    MP['lambda']=[1e-4,1e-4]

if __name__ == '__main__':
    mt.get_models(MP)
