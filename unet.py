from keras.models import *
from keras.preprocessing.image import array_to_img
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from unnecessary.data_processing_class import *


DATA_PATH = 'data'
IMG_WIDTH = 512
IMG_HEIGHT = 512
KERNEL_INITIALIZER = 'he_normal'
PADDING = 'same'
ACTIVATION = 'relu'
SMOOTH = 1
# K.set_image_dim_ordering('th')

class GetUnet(object):

    def __init__(self, img_height, img_width):
        self.img_height = img_height
        self.img_width = img_width

    def load_data(self):
        data = DataProcess(self.img_height, self.img_width)
        train_images, train_masks = data.load_train_data()
        test_images, test_masks = data.load_test_data()
        return train_images, train_masks, test_images, test_masks

    def dice_coef(self, y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        return (2. * intersection + smooth) / (union + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def dice_coef_2(self, y_true, y_pred, smooth=1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.dot(y_true_f, K.transpose(y_pred_f))
        union = K.dot(y_true_f, K.transpose(y_true_f)) + K.dot(y_pred_f, K.transpose(y_pred_f))
        return (2. * intersection + smooth) / (union + smooth)

    def dice_coef_loss_2(self, y_true, y_pred):
        return K.mean(1 - self.dice_coef_2(y_true, y_pred), axis=-1)

    def get_unet(self, activation, padding, kernel_initializer):
        inputs = Input((self.img_width, self.img_height, 1))

        '''
        unet with crop(because padding = valid) 
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
        print "conv1 shape:",conv1.shape
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
        print "conv1 shape:",conv1.shape
        crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
        print "crop1 shape:",crop1.shape
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print "pool1 shape:",pool1.shape
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
        print "conv2 shape:",conv2.shape
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
        print "conv2 shape:",conv2.shape
        crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
        print "crop2 shape:",crop2.shape
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print "pool2 shape:",pool2.shape
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
        print "conv3 shape:",conv3.shape
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
        print "conv3 shape:",conv3.shape
        crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
        print "crop3 shape:",crop3.shape
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print "pool3 shape:",pool3.shape
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)
        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)
        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)
        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
        '''
        ''' 
        conv1 = Conv2D(64, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(inputs)
        print('conv1 shape: {}'.format(conv1.shape))
        conv1 = Conv2D(64, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv1)
        print('conv1 shape: {}'.format(conv1.shape))
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print('pool1 shape:'.format(pool1.shape))

        conv2 = Conv2D(128, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(pool1)
        print('conv2 shape: {}'.format(conv2.shape))
        conv2 = Conv2D(128, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv2)
        print('conv2 shape: {}'.format(conv2.shape))
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print('pool2 shape:'.format(pool2.shape))

        conv3 = Conv2D(256, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(pool2)
        print('conv3 shape: {}'.format(conv3.shape))
        conv3 = Conv2D(256, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv3)
        print('conv3 shape: {}'.format(conv3.shape))
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print('pool3 shape:'.format(pool3.shape))

        conv4 = Conv2D(512, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(pool3)
        conv4 = Conv2D(512, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(pool4)
        conv5 = Conv2D(1024, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge6)
        conv6 = Conv2D(512, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv6)

        up7 = Conv2D(256, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge7)
        conv7 = Conv2D(256, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv7)

        up8 = Conv2D(128, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge8)
        conv8 = Conv2D(128, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv8)

        up9 = Conv2D(64, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode=activation, concat_axis=3)
        conv9 = Conv2D(64, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge9)
        conv9 = Conv2D(64, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv9)
        conv9 = Conv2D(2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        
        model = Model(input=inputs, output=conv10)
        '''

        conv1 = Conv2D(32, (3, 3), activation=activation, padding=padding, kernel_initializer=kernel_initializer)(inputs)
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

        # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=Adam(lr=1e-4), loss=self.dice_coef_loss, metrics=[self.dice_coef])
        # model.compile(optimizer=Adam(lr=1e-4), loss=self.dice_coef_loss_2, metrics=[self.dice_coef_2()])

        return model


    def train(self):
        print("Loading the data")
        train_images, train_masks, test_images, test_masks = self.load_data()
        print("Loading the data done")
        model = self.get_unet(ACTIVATION, PADDING, KERNEL_INITIALIZER)
        print("U-net received ")

        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(train_images,
                  train_masks,
                  batch_size=4,
                  epochs=10,
                  verbose=1,
                  validation_split=0.2,
                  shuffle=True,
                  callbacks=[model_checkpoint])

        print('Predicting test data')
        unet_test_masks = model.predict(test_images,
                                       batch_size=1,
                                       verbose=1)
        np.save(os.path.join(DATA_PATH, 'npy files', 'unet_test_masks.npy'), unet_test_masks)

    def save_images(self):
        print('Converting an array to image')
        imgs = np.load('unet_test_masks.npy')
        for i in range(imgs.shape[0]):
            img = imgs[i]
            img = array_to_img(img)
            img.save('data/test/unet_masks/%d.png' % i)

if __name__ == '__main__':
    unet = GetUnet(IMG_HEIGHT, IMG_WIDTH)
    unet.train()
    # unet.save_images()