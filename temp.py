from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import (Input,
                          Conv2D,
                          Conv2DTranspose,
                          concatenate,
                          MaxPooling2D)
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K


# K.set_image_dim_ordering('tf')
# K.set_image_data_format('channels_last')
IMG_WIDTH = 512
IMG_HEIGHT = 512

if K.image_data_format() == 'channels_first':
    input_shape = (1, IMG_WIDTH, IMG_HEIGHT)
else:
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 1)

datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest')

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
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, 1))
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

def train_and_predict():
    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    # image_datagen.fit(images, augment=True, seed=seed)
    # mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow_from_directory('data/train/images',
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        color_mode='grayscale',
                                                        class_mode=None,
                                                        seed=seed)

    mask_generator = mask_datagen.flow_from_directory('data/train/masks',
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      color_mode='grayscale',
                                                      class_mode=None,
                                                      seed=seed)

    # Combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    print('-'*40)
    print('Creating and compiling model...')
    print('-'*40)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=5)

    # model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,
    #           validation_split=0.2, callbacks=[model_checkpoint])

if __name__ == '__main__':
    train_and_predict()