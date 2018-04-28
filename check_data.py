import cv2
import numpy as np
from data_processing import DataProcess, IMG_HEIGHT, IMG_WIDTH
from train import COMPARE_WITH

IS_COMPARE = True           # for comparing unet and test datasets
if not IS_COMPARE:
    DATA_TYPE = 'unet'       # change to see the datasets (train, validation, test and unet)
FPS_RATE = 0.5

data = DataProcess(IMG_HEIGHT, IMG_WIDTH)
def get_images_and_masks(data_type):
    images, masks = data.load_data(data_type)
    return images, masks

if not IS_COMPARE:
    images, masks = get_images_and_masks(DATA_TYPE)
else:
    gtruth_images, gtruth_masks = get_images_and_masks(COMPARE_WITH)
    unet_images, unet_masks = get_images_and_masks('unet')

if not IS_COMPARE:
    for num_of_image in range(images.shape[0]):
        temp_image = images[num_of_image].reshape([data.img_height, data.img_width])
        temp_mask = masks[num_of_image].reshape([data.img_height, data.img_width])
        stacked_image = np.hstack((temp_image, temp_mask))
        cv2.imshow('Examples of test images', stacked_image)
        k = cv2.waitKey(round(1000 / FPS_RATE))                     # >0 - play, 0 - play with the pressing any key
        if k == 27:                                                 # wait for ESC key to exit
            break
        elif k == ord('s'):                                         # wait for 's' key to save and exit
            cv2.imwrite('data/%d_tumor.png' % num_of_image, stacked_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
else:
    for num_of_image in range(unet_images.shape[0]):

        temp_gtruth_image = gtruth_images[num_of_image].reshape([data.img_height, data.img_width])
        temp_gtruth_mask = gtruth_masks[num_of_image].reshape([data.img_height, data.img_width])
        stacked_image_1 = np.hstack((temp_gtruth_image, temp_gtruth_mask))

        temp_unet_image = unet_images[num_of_image].reshape([data.img_height, data.img_width])
        temp_unet_mask = unet_masks[num_of_image].reshape([data.img_height, data.img_width])
        stacked_image_2 = np.hstack((temp_unet_image, temp_unet_mask))

        stacked_image = np.vstack((stacked_image_1, stacked_image_2))

        cv2.imshow('Examples of test images', stacked_image)
        k = cv2.waitKey(round(1000 / FPS_RATE))                     # >0 - play, 0 - play with the pressing any key
        if k == 27:                                                 # wait for ESC key to exit
            break
        elif k == ord('s'):                                         # wait for 's' key to save and exit
            cv2.imwrite('data/%d_tumor.png' % num_of_image, stacked_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
