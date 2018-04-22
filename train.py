import keras
from keras.preprocessing.image import array_to_img
from keras.callbacks import ModelCheckpoint
from data_processing import *
from model.unet import *
from model.losses import *
import numpy as np
import tensorflow as tf
# cd /d C:\Program Files\NVIDIA Corporation\NVSMI\
# nvidia-smi.exe

DEVICE = 'cpu'
NUM_INTRA_THREADS = 4           # if device is 'CPU'
NUM_INTER_THREADS = 10          # if device is 'CPU'
GPU_MEMORY_FRACTION = 0.4       # if device is 'GPU'
DATA_PATH = 'data'
IMG_WIDTH = 512
IMG_HEIGHT = 512
# K.tensorflow_backend._get_available_gpus()

if DEVICE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_INTRA_THREADS,
                            inter_op_parallelism_threads=NUM_INTER_THREADS,
                            allow_soft_placement=True,
                            log_device_placement=True,
                            device_count={'CPU': 1})
    sess = tf.Session(config=config)
elif DEVICE == 'gpu':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    # set_session(tf.Session(config=config))
    sess = tf.Session(config=config)

class GetUnet(object):

    def __init__(self, img_height, img_width):
        self.__img_height = img_height
        self.__img_width = img_width

    def load_data(self):
        data = DataProcess(self.__img_height, self.__img_width)
        train_images, train_masks = data.load_train_data()
        test_images, test_masks = data.load_test_data()
        return train_images, train_masks, test_images, test_masks


    def get_unet(self):

        '''
        # def get_unet(self, activation, padding, kernel_initializer):
        inputs = Input((self.img_width, self.img_height, 1))
        conv1 = Conv2D(32, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(inputs)
        conv1 = Conv2D(32, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(pool1)
        conv2 = Conv2D(64, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # TODO: insert dropout layer
        conv3 = Conv2D(128, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(pool2)
        conv3 = Conv2D(128, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(pool3)
        conv4 = Conv2D(256, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(pool4)
        conv5 = Conv2D(512, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding=padding)(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(up6)
        conv6 = Conv2D(256, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding=padding, kernel_initializer=kernel_initializer)(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(up7)
        conv7 = Conv2D(128, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding=padding, kernel_initializer=kernel_initializer)(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(up8)
        conv8 = Conv2D(64, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding=padding)(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(up9)
        conv9 = Conv2D(32, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])

        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=[dice_coef])
        # model.compile(optimizer=Adam(lr=1e-4), loss=self.dice_coef_loss_2, metrics=[self.dice_coef_2()])
        '''

        model = get_unet_512(input_shape=(self.__img_height, self.__img_width, 1),
                             num_classes=1,
                             lr=0.0001,
                             loss=bce_dice_loss,
                             metrics=dice_coef)

        return model

    def train(self):
        print("Loading the data")
        train_images, train_masks, test_images, test_masks = self.load_data()
        print("Loading the data done")

        model = self.get_unet()
        model.summary()
        print("U-net received ")

        model_checkpoint = ModelCheckpoint(os.path.join('weights', 'unet.hdf5'),
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True)

        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs',
                                                  histogram_freq=0,
                                                  write_graph=True,
                                                  write_images=True)

        print('Fitting model...')
        history = model.fit(train_images,
                            train_masks,
                            batch_size=4,
                            epochs=2,
                            verbose=1,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=[tensorboard])

        print('Predicting test data')
        unet_test_masks = model.predict(test_images,
                                        batch_size=1,
                                        verbose=1)
        np.save(os.path.join(DATA_PATH, 'npy files', 'unet_test_masks.npy'), unet_test_masks)

    def save_images(self):
        print('\nConverting an array to image')
        images = np.load(os.path.join(DATA_PATH, 'npy files', 'unet_test_masks.npy'))
        # images = images.transpose(0, 3, 1, 2)
        for i in range(images.shape[0]):
            img = images[i]
            # img = Image.fromarray(img)
            # img = Image.fromarray(np.uint8(cmap=gist_earth(img) * 255))
            img = array_to_img(img)  # TODO: fix bug
            img.save(os.path.join(DATA_PATH, 'test', 'unet masks', '%03d_tumor_test.png') % (i + 1))
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, images.shape[0]))
            i += 1
            print('\nConverting done.')

if __name__ == '__main__':
    unet = GetUnet(IMG_HEIGHT, IMG_WIDTH)
    unet.train()
    unet.save_images()