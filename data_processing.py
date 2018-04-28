import os
import cv2
import time
import numpy as np
from sklearn import model_selection as sk

# Main settings
IMG_WIDTH = 128
IMG_HEIGHT = 128
IS_VISUALIZE = False
# NUM_IMAGES_TO_PROCESS = 200

# Additional settings
DATA_PATH = 'data'
DATA_TYPE = ('train', 'validation', 'test')
EXT = 'png'
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

class DataProcess(object):

    def __init__(self, img_height, img_width, data_path=DATA_PATH, img_ext=EXT):

        self.img_height = img_height
        self.img_width = img_width
        self.data_path = data_path
        self.img_ext = img_ext

    def create_data(self):

        images_path = os.path.join(self.data_path, 'images')
        masks_path = os.path.join(self.data_path, 'masks')

        # Check if files were created
        npy_list = []
        is_npy_exist = True
        for i in range(len(DATA_TYPE)):
            npy_list.append(DATA_TYPE[i] + '_images.npy')
            npy_list.append(DATA_TYPE[i] + '_masks.npy')
            is_images_exist = os.path.exists(os.path.join(self.data_path, 'npy files', npy_list[i]))
            is_masks_exist = os.path.exists(os.path.join(self.data_path, 'npy files', npy_list[i + 1]))
            is_npy_exist = is_npy_exist * (is_images_exist or is_masks_exist)

        if not is_npy_exist:
            images_list = os.listdir(images_path)
            images_list.sort()
            try:
                num_samples = NUM_IMAGES_TO_PROCESS
            except NameError:
                print("\nVariable 'NUM_IMAGES_TO_PROCESS' was not set.\nFull set of images will be processed.")
                num_samples = int(len(images_list))

            images = np.ndarray((num_samples, self.img_height, self.img_width, 1), dtype=np.uint8)
            masks = np.ndarray((num_samples, self.img_height, self.img_width, 1), dtype=np.uint8)

            start = time.time()
            print('-' * 30)
            print('\tCreating the data...')
            print('\tNumber of images: {}'.format(num_samples))
            print('-' * 30)

            for i in range(num_samples):
                image_name = images_list[i]
                if 'mask' in image_name:
                    continue
                mask_name = image_name.split('.')[0] + '_mask.' + self.img_ext
                image = cv2.imread(os.path.join(images_path, image_name), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(os.path.join(masks_path, mask_name), cv2.IMREAD_GRAYSCALE)

                if image.shape[0] != self.img_height or image.shape[1] != self.img_width:
                    image, mask = self.image_rescale(image, mask)

                image = np.array([image])
                mask = np.array([mask])
                image = image.transpose(1, 2, 0)
                mask = mask.transpose(1, 2, 0)
                images[i] = image
                masks[i] = mask

                if IS_VISUALIZE:
                    temp_image = images[i].reshape([self.img_height, self.img_width])
                    temp_mask = masks[i].reshape([self.img_height, self.img_width])
                    stacked_image = np.hstack((temp_image, temp_mask))
                    cv2.imshow('Source image and its mask', stacked_image)
                    cv2.waitKey(1000)

                if (i+1) % 100 == 0:
                        print('Done: {0}/{1} images'.format((i+1), num_samples))
            end = time.time()
            print('Creating the data done ({:1.2f} seconds)'.format(end - start))

            train_images, test_images, train_masks, test_masks = sk.train_test_split(images,
                                                                                     masks,
                                                                                     shuffle=True,
                                                                                     test_size=TEST_RATIO,
                                                                                     random_state=1)

            train_images, val_images, train_masks, val_masks = sk.train_test_split(train_images,
                                                                                   train_masks,
                                                                                   shuffle=True,
                                                                                   test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO),
                                                                                   random_state=1)

            start = time.time()
            np.save(os.path.join(self.data_path, 'npy files', DATA_TYPE[0] + '_images.npy'), train_images)
            np.save(os.path.join(self.data_path, 'npy files', DATA_TYPE[1] + '_images.npy'), val_images)
            np.save(os.path.join(self.data_path, 'npy files', DATA_TYPE[2] + '_images.npy'), test_images)

            np.save(os.path.join(self.data_path, 'npy files', DATA_TYPE[0] + '_masks.npy'), train_masks)
            np.save(os.path.join(self.data_path, 'npy files', DATA_TYPE[1] + '_masks.npy'), val_masks)
            np.save(os.path.join(self.data_path, 'npy files', DATA_TYPE[2] + '_masks.npy'), test_masks)
            end = time.time()
            print('Saving to .npy files done ({:1.2f} seconds)'.format(end - start))
        else:
            print('-' * 38)
            print('\tDatasets were already created.')
            print('-' * 38)

    def image_rescale(self, image, mask, interpolation=cv2.INTER_CUBIC):
        output_image = cv2.resize(image, (self.img_height, self.img_width), interpolation=interpolation)
        output_mask = cv2.resize(mask, (self.img_height, self.img_width), interpolation=interpolation)
        return output_image, output_mask

    def load_data(self, data_type):
        loaded_images = np.load(os.path.join(self.data_path, 'npy files', data_type + '_images.npy'))
        loaded_masks = np.load(os.path.join(self.data_path, 'npy files', data_type + '_masks.npy'))
        loaded_images = loaded_images.astype('float32')
        loaded_masks = loaded_masks.astype('float32')
        loaded_images /= 255
        loaded_masks /= 255
        return loaded_images, loaded_masks                    # output dtype: uint8 (0-1)

if __name__ == '__main__':
    data = DataProcess(IMG_HEIGHT, IMG_WIDTH)
    data.create_data()
    # cProfile.run('data.create_data()')
    # data.load_data('train')