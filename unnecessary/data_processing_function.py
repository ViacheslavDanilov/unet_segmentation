import os
import cv2
import numpy as np

data_type = 'test'
DATA_PATH = 'data/'
IMAGES_PATH = os.path.join(DATA_PATH, data_type, 'images/tumor')
MASKS_PATH = os.path.join(DATA_PATH, data_type, 'masks/tumor')
EXT = 'png'
IMG_WIDTH = 512
IMG_HEIGHT = 512

def create_data(data_type):

    is_exists_images_path = os.path.exists(os.path.join(DATA_PATH, 'npy files', data_type + '_images.npy'))
    is_exists_masks_path = os.path.exists(os.path.join(DATA_PATH, 'npy files', data_type + '_masks.npy'))
    if not is_exists_images_path or not is_exists_masks_path:
        i = 0
        images_list = os.listdir(IMAGES_PATH)
        num_samples = int(len(images_list))
        images = np.ndarray((num_samples, 1, IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)
        masks = np.ndarray((num_samples, 1, IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)

        print('-' * 40)
        print('Creating training images...')
        print('-' * 40)

        for image_name in images_list:
            if 'mask' in image_name:
                continue
            mask_name = image_name.split('.')[0] + '_mask.'+ EXT
            image = cv2.imread(os.path.join(IMAGES_PATH, image_name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(os.path.join(MASKS_PATH, mask_name), cv2.IMREAD_GRAYSCALE)

            if image.shape[0] != IMG_HEIGHT or image.shape[0] != IMG_WIDTH:
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)

            image = np.array([image])
            mask = np.array([mask])

            images[i] = image
            masks[i] = mask

            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, num_samples))
            i += 1
        print('Loading done.')

        np.save(os.path.join(DATA_PATH, 'npy files', data_type + '_images.npy'), images)
        np.save(os.path.join(DATA_PATH, 'npy files', data_type + '_masks.npy'), masks)
        print('Saving to .npy files done.')
    else:
        print("\nDatasets '{0}_images.npy' and '{0}_masks.npy' already created.".format(data_type))

def load_train_data():
    images_train = np.load(os.path.join(DATA_PATH, 'npy files', 'train_images.npy'))
    masks_train = np.load(os.path.join(DATA_PATH, 'npy files', 'train_masks.npy'))
    return images_train, masks_train

def load_test_data():
    images_test = np.load(os.path.join(DATA_PATH, 'npy files', 'test_images.npy'))
    masks_test = np.load(os.path.join(DATA_PATH, 'npy files', 'test_masks.npy'))
    return images_test, masks_test

if __name__ == '__main__':
    create_data(data_type)
    # load_train_data()
    # load_test_data()
    print('Done!')