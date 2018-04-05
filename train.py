from __future__ import print_function

import os
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import (Input,
                          Conv2D,
                          Conv2DTranspose,
                          concatenate,
                          MaxPooling2D)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

# K.set_image_data_format('channels_last')

data_path = 'data/'
save_path = 'save/'
image_rows = 512
image_cols = 512

def create_train_data():
    i = 0
    train_data_path = os.path.join(data_path, 'train')
    images_list = os.listdir(train_data_path)
    number_of_samples = int(len(images_list)/2)
    images = np.ndarray((number_of_samples, 1, image_rows, image_cols), dtype=np.uint8)
    masks = np.ndarray((number_of_samples, 1, image_rows, image_cols), dtype=np.uint8)

    print('-'*40)
    print('Creating training images...')
    print('-'*40)

    for image_name in images_list:
        if 'mask' in image_name:
            continue
        mask_name = image_name.split('.')[0]+'_mask.tif'
        image = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(train_data_path, mask_name), cv2.IMREAD_GRAYSCALE)

        image = np.array([image])
        mask = np.array([mask])

        images[i] = image
        masks[i] = mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, number_of_samples))
        i += 1
    print('Loading done.')

    images_path = Path(os.path.join(data_path, 'npy files'), 'images_train.npy')
    masks_path = Path(os.path.join(data_path, 'npy files'), 'masks_train.npy')
    if images_path.is_file() == False:
        np.save(os.path.join(data_path, 'npy files', 'images_train.npy'), images)
        np.save(os.path.join(data_path, 'npy files', 'masks_train.npy'), masks)
        print('Saving to .npy files done.')
    else:
        print('images_train.npy and masks_train.npy already created.')

def load_train_data():
    images_train = np.load('images_train.npy')
    masks_train = np.load('masks_train.npy')
    return images_train, masks_train



def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_coef_2(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.dot(y_true, K.transpose(y_pred))
    union = K.dot(y_true, K.transpose(y_true))+K.dot(y_pred, K.transpose(y_pred))
    return (2. * intersection + smooth) / (union + smooth)

def dice_coef_loss_2(y_true, y_pred):
    return K.mean(1 - dice_coef(y_true, y_pred), axis=-1)

def get_unet():
    inputs = Input((image_rows, image_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

if __name__ == '__main__':
    create_train_data()
    # load_train_data()
    # create_test_data()