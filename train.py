import glob
import platform
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.utils import plot_model
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from data_processing import *
from utils.unet import get_unet_128, get_unet_256, get_unet_512, get_unet_1024
from utils.losses import dice_coef, jaccard_coef, dice_loss, jaccard_loss, bce_dice_loss

# cd /d C:\Program Files\NVIDIA Corporation\NVSMI\
# nvidia-smi.exe
# tensorboard --logdir=model/logs

# Main settings
DEVICE = 'gpu'                              # choose 'cpu' or 'gpu' (BATCH SIZE = 2, WIDTH and HEIGHT = 128)
EPOCHS = 1
BATCH_SIZE = 1
LOSS = bce_dice_loss                        # 'binary_crossentropy'
COMPARE_WITH = 'test'

# Additional settings
THRESHOLD_LEVEL = 0.05                      # threshold for output image of unet (for obtaining BW image/mask)
DROP_PROB = 0.5
OPTIMIZER = Adam
LR = 0.001
METRIC = [dice_coef, jaccard_coef]
KERNEL_INITIALIZER = 'he_normal'
KERNEL_REGULARIZER = None                   # set only if videomemory > 6 GB
NUM_INTRA_THREADS = 4                       # if device is 'cpu'
NUM_INTER_THREADS = 8                       # if device is 'cpu'
GPU_MEMORY_FRACTION = 1.0                   # if device is 'gpu'

if DEVICE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"            # IDs match nvidia-smi
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"                  # "0, 1" for multiple
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_INTRA_THREADS,
                            inter_op_parallelism_threads=NUM_INTER_THREADS,
                            allow_soft_placement=True,
                            log_device_placement=True,
                            device_count={'CPU': 1})
    sess = tf.Session(config=config)
elif DEVICE == 'gpu':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
    sess = tf.Session(config=config)

class GetUnet(object):

    def __init__(self, img_height, img_width, data_path=DATA_PATH):
        self.img_height = img_height
        self.img_width = img_width
        self.data_path = data_path

    def get_unet(self, action):
        unet_action_map = {'128': get_unet_128,
                           '256': get_unet_256,
                           '512': get_unet_512,
                           '1024': get_unet_1024}

        model = unet_action_map[action](input_shape=(self.img_height, self.img_width, 1),
                                        num_classes=1,
                                        drop_prob=DROP_PROB,
                                        kernel_initializer=KERNEL_INITIALIZER,
                                        kernel_regularizer=KERNEL_REGULARIZER,
                                        optimizer=OPTIMIZER,
                                        lr=LR,
                                        loss=LOSS,
                                        metrics=METRIC)

        return model

    def get_augmented_data(self, images, masks, num_transforms):

        if num_transforms == 0:
            aug_train_images = images
            aug_train_masks = masks
        else:
            image_datagen = ImageDataGenerator(rescale=1./255,
                                               rotation_range=90,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               shear_range=10,
                                               zoom_range=[0.8, 1.2],
                                               brightness_range=(0.5, 1.5),
                                               horizontal_flip=True,
                                               vertical_flip=True,
                                               fill_mode='nearest')

            mask_datagen = ImageDataGenerator(rotation_range=90,
                                              width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              shear_range=10,
                                              zoom_range=[0.8, 1.2],
                                              brightness_range=None,
                                              horizontal_flip=True,
                                              vertical_flip=True,
                                              fill_mode='nearest')

            image_datagen.fit(images)
            mask_datagen.fit(masks)
            # val_datagen.fit(val_images)
            # val_datagen.fit(val_masks)

            seed = 11
            aug_train_images = np.ndarray((num_transforms * images.shape[0], self.img_height, self.img_width, 1), dtype=np.float32)
            aug_train_masks = np.ndarray((num_transforms * masks.shape[0], self.img_height, self.img_width, 1), dtype=np.float32)

            # Augmentation of images
            idx = 0
            for count in range(images.shape[0]):
                temp_image = images[count]
                temp_image = np.expand_dims(temp_image, axis=0)
                for batch in image_datagen.flow(temp_image,
                                                y=None,
                                                batch_size=1,
                                                shuffle=True,
                                                seed=seed):
                    aug_train_images[count] = batch
                    idx += 1
                    if idx > num_transforms:
                        break  # otherwise the generator would loop indefinitely

            # Augmentation of masks

            idx = 0
            for count in range(masks.shape[0]):
                temp_mask = masks[count]
                temp_mask = np.expand_dims(temp_mask, axis=0)
                for batch in mask_datagen.flow(temp_mask,
                                               y=None,
                                               batch_size=1,
                                               shuffle=True,
                                               seed=seed):
                    aug_train_masks[count] = batch
                    idx += 1
                    if idx > num_transforms:
                        break  # otherwise the generator would loop indefinitely

        return aug_train_images, aug_train_masks

    def train(self):

        # Loading the datasets
        data = DataProcess(self.img_height, self.img_width)
        train_images, train_masks = data.load_data('train')
        val_images, val_masks = data.load_data('validation')

        # Getting the model of u-net
        print('Receiving the U-net...')
        start = time.time()
        model = self.get_unet(str(self.img_height))
        model.summary()
        end = time.time()
        print('Receiving the U-net done ({:1.2f} seconds)'.format(end - start))

        # Delete old files
        self.clean_folder('model/logs')
        self.clean_folder('model/hdf5 models')

        aug_train_images, aug_train_masks = self.get_augmented_data(train_images, train_masks, 0)
        ################################################################################################################
        # Visualize 
        # idx = 0
        # while True:
        #     temp_image = aug_train_images[idx].reshape([data.img_height, data.img_width])
        #     temp_mask = aug_train_masks[idx].reshape([data.img_height, data.img_width])
        #     stacked_image = np.hstack((temp_image, temp_mask))
        #     cv2.imshow('Image', stacked_image)
        #     k = cv2.waitKey(3000)
        #     idx += 1
        #     if idx > 30:
        #         break                       # otherwise the generator would loop indefinitely

        ################################################################################################################

        # Fitting the model
        print('Fitting the model...')
        start = time.time()
        callbacks = [TensorBoard(log_dir='./model/logs',
                                 histogram_freq=0,
                                 batch_size=BATCH_SIZE,
                                 write_graph=True,
                                 write_grads=True,
                                 write_images=True),
                    EarlyStopping(monitor='val_dice_coef',
                                  min_delta=0.0001,
                                  patience=10,
                                  verbose=1,
                                  mode='max'),
                    ModelCheckpoint('model/hdf5 models/model.{epoch:02d}-{dice_coef:.2f}-{val_dice_coef:.2f}'
                                    '-{jaccard_coef:.2f}-{val_jaccard_coef:.2f}.hdf5 models',
                                    monitor='val_dice_coef',
                                    save_best_only=True,
                                    save_weights_only=False,
                                    verbose=1),
                    CSVLogger('model/history.csv',
                              separator=',',
                              append=False),
                    ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.2,
                                      patience=5,
                                      verbose=1,
                                      min_lr=0.0001)]

        # history = model.fit_generator(datagen.flow(train_images, train_masks, batch_size=BATCH_SIZE),
        #                               steps_per_epoch=train_images.shape[0]//BATCH_SIZE,
        #                               epochs=EPOCHS,
        #                               verbose=1,
        #                               callbacks=callbacks,
        #                               validation_data=datagen.flow(val_images, val_masks, batch_size=BATCH_SIZE),
        #                               validation_steps=val_images.shape[0]//BATCH_SIZE,
        #                               shuffle=True)
        history = model.fit(x=aug_train_images,
                            y=aug_train_masks,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=(val_images, val_masks),
                            shuffle=True)


        # history = model.fit(x=train_images,
        #                     y=train_masks,
        #                     batch_size=BATCH_SIZE,
        #                     epochs=EPOCHS,
        #                     verbose=1,
        #                     callbacks=callbacks,
        #                     validation_data=(val_images, val_masks),
        #                     shuffle=True)
        end = time.time()
        print('Fitting the model done ({:1.2f} seconds)'.format(end - start))

        self.save_model(model)

    def save_model(self, model):
        print('Saving the model and its model...')
        start = time.time()
        if platform.system() == 'Linux':
            plot_model(model, to_file='model/unet_model.png', show_shapes=True, show_layer_names=True)
        model.save_weights('model/unet_weights.h5', overwrite=True)
        model_json = model.to_json()
        json_file = open('model/unet_architecture.json', 'w')
        json_file.write(model_json)
        json_file.close()
        end = time.time()
        print('Saving the model and its model ({:1.2f} seconds)'.format(end - start))

    def load_model(self):
        print('Loading the model and its model...')
        start = time.time()
        json_file = open('model/unet_architecture.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('model/unet_weights.h5')
        optimizer = OPTIMIZER(lr=LR)
        loaded_model.compile(optimizer=optimizer, loss=LOSS, metrics=METRIC)
        end = time.time()
        print('Loading the model and its model ({:1.2f} seconds)'.format(end - start))
        return loaded_model

    def evaluate_model(self, threshold_level, compare_with):
        data = DataProcess(self.img_height, self.img_width)
        loaded_model = self.load_model()

        images_to_compare, masks_to_compare = data.load_data(compare_with)
        print('Predicting the ' + compare_with + ' data...')
        unet_images = loaded_model.predict(images_to_compare,
                                           batch_size=BATCH_SIZE,
                                           verbose=1)
        unet_masks = (unet_images > threshold_level).astype(np.uint8)
        unet_images *= 255
        unet_masks *= 255
        score = loaded_model.evaluate(unet_masks, masks_to_compare, batch_size=BATCH_SIZE, verbose=1)
        print('Accuracy for ' + compare_with + ' data: %.2f%%' % (score[1] * 100))

        np.save(os.path.join(DATA_PATH, 'npy files', 'unet_images.npy'), unet_images)
        np.save(os.path.join(DATA_PATH, 'npy files', 'unet_masks.npy'), unet_masks)

    def save_images(self):
        self.clean_folder('output/test images')
        self.clean_folder('output/test masks')
        self.clean_folder('output/unet images')
        self.clean_folder('output/unet masks')
        print('\nConverting the array to images...')
        start = time.time()
        gtruth_images = np.load(os.path.join(DATA_PATH, 'npy files', 'test_images.npy'))
        gtruth_masks = np.load(os.path.join(DATA_PATH, 'npy files', 'test_masks.npy'))
        unet_images = np.load(os.path.join(DATA_PATH, 'npy files', 'unet_images.npy'))
        unet_masks = np.load(os.path.join(DATA_PATH, 'npy files', 'unet_masks.npy'))
        for i in range(unet_images.shape[0]):
            gtruth_image = gtruth_images[i]
            gtruth_mask = gtruth_masks[i]
            unet_image = unet_images[i]
            unet_mask = unet_masks[i]

            gtruth_image = array_to_img(gtruth_image)
            gtruth_mask = array_to_img(gtruth_mask)
            unet_image = array_to_img(unet_image)
            unet_mask = array_to_img(unet_mask)

            gtruth_image.save(os.path.join('output', 'ground truth images', '%04d_test_image.png') % (i + 1))
            gtruth_mask.save(os.path.join('output', 'ground truth masks', '%04d_test_mask.png') % (i + 1))
            unet_image.save(os.path.join('output', 'unet images', '%04d_unet_image.png') % (i + 1))
            unet_mask.save(os.path.join('output', 'unet masks', '%04d_unet_mask.png') % (i + 1))
            if (i+1) % 50 == 0:
                print('\tDone: {0}/{1} images'.format((i+1), unet_images.shape[0]))
            i += 1
        end = time.time()
        print('Converting the array to images done ({:1.2f} seconds)'.format(end - start))

    def clean_folder(self, path):
        files = glob.glob(path + '/*')
        for f in files:
            try:
                os.remove(f)
            except OSError:
                print('Cannot delete file. Make sure your have enough credentials '
                      'to delete this file or that no other process is using this file.')

if __name__ == '__main__':
    unet = GetUnet(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    # unet.train()
    unet.evaluate_model(threshold_level=THRESHOLD_LEVEL, compare_with=COMPARE_WITH)
    # unet.save_images()
