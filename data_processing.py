import os
import cv2
import numpy as np

DATA_TYPE = 'test'
DATA_PATH = 'data'
EXT = 'png'
IMG_WIDTH = 512
IMG_HEIGHT = 512
IS_VISUALIZE = False
# NUM_IMAGES_TO_PROCESS = 20

class DataProcess(object):

    def __init__(self, img_height, img_width, data_path=DATA_PATH, data_type=DATA_TYPE, img_ext=EXT):

        self.__img_height = img_height
        self.__img_width = img_width
        self.__data_path = data_path
        self.__data_type = data_type
        self.__img_ext = img_ext

    def create_data(self):

        images_path = os.path.join(self.__data_path, self.__data_type, 'images/tumor')
        masks_path = os.path.join(self.__data_path, self.__data_type, 'masks/tumor')
        is_exists_images_path = os.path.exists(os.path.join(self.__data_path, 'npy files', self.__data_type + '_images.npy'))
        is_exists_masks_path = os.path.exists(os.path.join(self.__data_path, 'npy files', self.__data_type + '_masks.npy'))
        if not is_exists_images_path or not is_exists_masks_path:
            i = 0
            images_list = os.listdir(images_path)
            try:
                num_samples = NUM_IMAGES_TO_PROCESS
            except NameError:
                print("Variable 'NUM_IMAGES_TO_PROCESS' not set. Full set of images will be processed.")
                num_samples = int(len(images_list))

            images = np.ndarray((num_samples, self.__img_height, self.__img_width, 1), dtype=np.uint8)
            masks = np.ndarray((num_samples, self.__img_height, self.__img_width, 1), dtype=np.uint8)

            print('-' * 27)
            print('Creating ' + self.__data_type + 'ing images...')
            print('Number of images: {}'.format(num_samples))
            print('-' * 27)

            for image_name in images_list:
                while i < num_samples:
                    if 'mask' in image_name:
                        continue
                    mask_name = image_name.split('.')[0] + '_mask.' + self.__img_ext
                    image = cv2.imread(os.path.join(images_path, image_name), cv2.IMREAD_GRAYSCALE)
                    mask = cv2.imread(os.path.join(masks_path, mask_name), cv2.IMREAD_GRAYSCALE)

                    if image.shape[0] != self.__img_height or image.shape[1] != self.__img_width:
                        image, mask = self.image_rescale(image, mask)

                    image = np.array([image])
                    mask = np.array([mask])
                    image = image.transpose(1, 2, 0)
                    mask = mask.transpose(1, 2, 0)
                    images[i] = image
                    masks[i] = mask

                    if IS_VISUALIZE:
                        temp_image = images[i].reshape([512, 512])
                        temp_mask = masks[i].reshape([512, 512])
                        stacked_image = np.hstack((temp_image, temp_mask))
                        cv2.imshow('Source image and its mask', stacked_image)
                        cv2.waitKey(2000)

                    if i % 100 == 0:
                        print('Done: {0}/{1} images'.format(i, num_samples))
                    i += 1
                else:
                    break
            print('\nLoading done.')

            np.save(os.path.join(self.__data_path, 'npy files', self.__data_type + '_images.npy'), images)
            np.save(os.path.join(self.__data_path, 'npy files', self.__data_type + '_masks.npy'), masks)
            print('Saving to .npy files done.')
        else:
            print('-' * 66)
            print("Datasets '{0}_images.npy' and '{0}_masks.npy' already created.".format(self.__data_type))
            print('-' * 66)

    def image_rescale(self, image, mask, interpolation=cv2.INTER_CUBIC):
        output_image = cv2.resize(image, (self.__img_height, self.__img_width), interpolation=interpolation)
        output_mask = cv2.resize(mask, (self.__img_height, self.__img_width), interpolation=interpolation)
        return output_image, output_mask

    def load_train_data(self):
        train_images = np.load(os.path.join(self.__data_path, 'npy files', 'train_images.npy'))
        train_masks = np.load(os.path.join(self.__data_path, 'npy files', 'train_masks.npy'))
        train_images = train_images.astype('float32')
        train_masks = train_masks.astype('float32')
        train_images /= 255
        train_masks /= 255
        return train_images, train_masks

    def load_test_data(self):
        test_images = np.load(os.path.join(self.__data_path, 'npy files', 'test_images.npy'))
        test_masks = np.load(os.path.join(self.__data_path, 'npy files', 'test_masks.npy'))
        test_images = test_images.astype('float32')
        test_masks = test_masks.astype('float32')
        test_images /= 255
        test_masks /= 255
        return test_images, test_masks

if __name__ == '__main__':
    data = DataProcess(IMG_HEIGHT, IMG_WIDTH)
    data.create_data()
    # data.load_train_data()
    # data.load_test_data()