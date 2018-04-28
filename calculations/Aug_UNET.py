import os
import sys
import random

import shutil
import warnings
import argparse
import datetime

# Set some arg params

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epochs', default=60, type=int)
args = parser.parse_args()
seed = 42
random.seed = seed

import numpy as np
np.random.seed = seed

import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
import skimage.io
from skimage import transform, util, exposure
from skimage.filters import gaussian
from skimage.transform import resize, rescale
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist

from keras.models import Model, load_model
from keras.layers import Input, Activation
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
tf.set_random_seed(seed)

# Set some parameters
currentDT = datetime.datetime.now()
day = currentDT.day
hr = currentDT.hour
mn = currentDT.minute
name = 'Sub_' + str(day) + '_' + str(hr) + '_' + str(mn) + '_BN'
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'data/train/'
TEST_PATH = 'data/test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')




train_ids = next(os.walk(TRAIN_PATH))[1]
# print(train_ids)
test_ids = next(os.walk(TEST_PATH))[1]
# print(test_ids)

# train_ids = train_ids[:5]
# test_ids = test_ids[:5]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')


def augmentation(scans, masks, n):
    data_gen_args = dict(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=90,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[0.9, 1.2])
    datagen = ImageDataGenerator(**data_gen_args)
    # mask_datagen = ImageDataGenerator(**data_gen_args)
    # datagen.fit(scans)
    i = 0
    scans_g = scans.copy()
    for batch in datagen.flow(scans, batch_size=1):
        scans_g = np.vstack([scans_g, batch])
        i += 1
        if i > n:
            break
    i = 0
    masks_g = masks.copy()
    for batch in datagen.flow(masks, batch_size=1):
        masks_g = np.vstack([masks_g, batch])
        i += 1
        if i > n:
            break
    return scans_g, masks_g
    
def data_aug(image, masked, angle=70, resize_rate=1):
    new_image = image.copy()
    new_masked = masked.copy()
    flip = random.randint(0, 1)
    # blur = random.randint(0, 1)
    # invert = random.randint(0, 1)
    # expos = random.randint(0, 1)
    # sh = random.random() / 2 - 0.25
    sh = 0
    rotate_angle = random.random() / 180 * np.pi * angle
    # Create Affine transform
    affine_tf = transform.AffineTransform(shear=sh, rotation=rotate_angle)
    # Apply transform to new_image data
    new_image = transform.warp(new_image, inverse_map=affine_tf, mode='edge')
    new_masked = transform.warp(new_masked, inverse_map=affine_tf, mode='edge')
    # Ramdomly change the exposure
    # if expos:
        # new_image = exposure.rescale_intensity(new_image)
    # Ramdomly invert the colors
    # if invert:
        # new_image = util.invert(new_image)
    # Ramdomly flip frame
    if flip:
        new_image = new_image[:, ::-1, :]
        new_masked = new_masked[:, ::-1]
    # if blur:
        # new_image = gaussian(new_image, sigma=2, mode='reflect')
    new_image = np.expand_dims(new_image, axis=0)
    new_masked = np.expand_dims(new_masked, axis=0)
    return new_image, new_masked

    #
    # plt.imshow(scans_g[0, :])
    # plt.show()
    # lol = masks_g[0, :].squeeze()
    # plt.imshow(lol)
    # plt.show()
    # plt.imshow(scans_g[len(scans_g)-1, :])
    # plt.show()
    # lol = masks_g[len(scans_g)-1, :].squeeze()
    # plt.imshow(lol)
    # plt.show()
    # return ((scans_g, masks_g))


def acquire(X_train, Y_train, N=None):
    X = X_train.copy()
    Y = Y_train.copy()
    if not N:
        N = int(.8*len(X_train))
    victims = np.random.choice(len(X_train), N)
    for num in victims:
        (image, masked) = data_aug(X[num], Y[num], angle=180, resize_rate=1)
        X = np.vstack([X, image])
        Y = np.vstack([Y, masked])
    permutations = np.random.permutation(len(X))
    X = X[permutations]
    Y = Y[permutations]
    return X, Y

xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=seed)
X_aug, Y_aug = acquire(xtr, ytr)


permutations = np.random.permutation(len(X_aug))
X_aug = X_aug[permutations]
Y_aug = Y_aug[permutations]

def generator(X_train, X_val, Y_train, Y_val, batch_size):
    data_gen_args = dict(featurewise_center=False,
                         samplewise_center=False,
                         featurewise_std_normalization=False,
                         samplewise_std_normalization=False,
                         zca_whitening=False,
                         rotation_range=90,
                         width_shift_range=0.3,
                         height_shift_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True,
                         zoom_range=[0.9, 1.2])
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_datagen.fit(X_train, augment=True)
    mask_datagen.fit(Y_train, augment=True)
    image_generator = image_datagen.flow(X_train, batch_size=batch_size)
    mask_generator = mask_datagen.flow(Y_train, batch_size=batch_size)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(X_val)
    mask_datagen_val.fit(Y_val)
    image_generator_val = image_datagen_val.flow(X_val, batch_size=batch_size)
    mask_generator_val = mask_datagen_val.flow(Y_val, batch_size=batch_size)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

   
    

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

    
    

# data_aug(X_train[0], Y_train[0], angel=30, resize_rate=0.9)
# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255)(inputs)

# here through comments I wrote original parameters of U-net

c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(s)  # (64, (3, 3)) with relu
c1 = BatchNormalization()(c1)  # None
c1 = Dropout(0.1)(c1)  # None
c1 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c1)  # (64, (3, 3)) with relu
c1 = BatchNormalization()(c1)  # None
p1 = MaxPooling2D((2, 2))(c1)
p1 = Activation('elu')(p1)

c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(p1)
c2 = BatchNormalization()(c2)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c2)
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2))(c2)
p2 = Activation('elu')(p2)

c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(p2)
c3 = BatchNormalization()(c3)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c3)
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2, 2))(c3)
p3 = Activation('elu')(p3)

c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(p3)
c4 = BatchNormalization()(c4)
c4 = Dropout(0.3)(c4)
c4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c4)
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)
p4 = Activation('elu')(p4)

c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(p4)
c5 = BatchNormalization()(c5)
c5 = Dropout(0.5)(c5)
c5 = Conv2D(256, (3, 3), kernel_initializer='he_normal', padding='same')(c5)
c5 = BatchNormalization()(c5)
c5 = Activation('elu')(c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = BatchNormalization()(u6)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(u6)
c6 = BatchNormalization()(c6)
c6 = Dropout(0.3)(c6)
c6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', padding='same')(c6)
c6 = Activation('elu')(c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = BatchNormalization()(u7)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(u7)
c7 = BatchNormalization()(c7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(c7)
c7 = BatchNormalization()(c7)
c7 = Activation('elu')(c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = BatchNormalization()(u8)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(u8)
c8 = BatchNormalization()(c8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), kernel_initializer='he_normal', padding='same')(c8)
c8 = BatchNormalization()(c8)
c8 = Activation('elu')(c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = BatchNormalization()(u9)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(u9)
c9 = BatchNormalization()(c9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), kernel_initializer='he_normal', padding='same')(c9)
c9 = BatchNormalization()(c9)
c9 = Activation('elu')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()


# Fit model
earlystopper = EarlyStopping(patience=15, verbose=1)
checkpointer = ModelCheckpoint(name + '.h5', verbose=1, save_best_only=True)
# scans_g, masks_g = augmentation(X_train, Y_train, 150)

# permutations = np.random.permutation(len(scans_g))
# scans_g = scans_g[permutations]
# masks_g = masks_g[permutations]
# xtr, xval, ytr, yval = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
# train_generator, val_generator = generator(xtr, xval, ytr, yval, batch_size=args.batch_size)
# results = model.fit_generator(train_generator, steps_per_epoch=len(xtr)/args.batch_size, epochs=args.epochs,  validation_data=val_generator, shuffle=True, validation_steps=len(xval)/args.batch_size, callbacks=[earlystopper, checkpointer])

# Predict on train, val and test


results = model.fit(X_aug, Y_aug, validation_data=(xval, yval), shuffle=True, batch_size=args.batch_size,
                    epochs=args.epochs, callbacks=[earlystopper, checkpointer])

model = load_model(name + '.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(
        resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(name + '.csv', index=False)
