import os
import cv2
import random
from keras.preprocessing.image import ImageDataGenerator

FPS_RATE = 1
is_visualize = True
NUM_AUGMENTED_IMAGES = 20
NUM_IMAGE = random.randint(1, 1532)
SAVE_PATH = 'augmentation'
SOURCE_PATH = 'data/images/' + str(NUM_IMAGE).zfill(4) + '_tumor.png'
GROUND_TRUTH_PATH = 'data/masks/' + str(NUM_IMAGE).zfill(4) + '_tumor_mask.png'

# Remove old images
file_list = os.listdir(SAVE_PATH)
for file_name in file_list:
    os.remove(SAVE_PATH + '/' + file_name)

# Augmentation
datagen_args = dict(rotation_range=90,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=10,
                    zoom_range=[0.8, 1.2],
                    brightness_range=(0.5, 1.5),
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

datagen = ImageDataGenerator(**datagen_args)

img = cv2.imread(SOURCE_PATH, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(GROUND_TRUTH_PATH, cv2.IMREAD_GRAYSCALE)
img = img.reshape([1, img.shape[0], img.shape[1], 1])
mask = mask.reshape([1, mask.shape[0], mask.shape[1], 1])


# Images
seed = 11
i = 0
for batch in datagen.flow(img,
                          y=None,
                          batch_size=1,
                          shuffle=True,
                          save_to_dir=SAVE_PATH,
                          save_prefix='tumor',
                          save_format='png',
                          seed=seed):
    i += 1
    if i > NUM_AUGMENTED_IMAGES-1:
        break  # otherwise the generator would loop indefinitely
# Masks
i = 0
for batch in datagen.flow(mask,
                          y=None,
                          batch_size=1,
                          shuffle=True,
                          save_to_dir=SAVE_PATH,
                          save_prefix='tumor_mask',
                          save_format='png',
                          seed=seed):
    i += 1
    if i > NUM_AUGMENTED_IMAGES-1:
        break  # otherwise the generator would loop indefinitely

if is_visualize:
    aug_images_list = os.listdir(SAVE_PATH)
    aug_images_list.sort(reverse=True)
    for idx in range(NUM_AUGMENTED_IMAGES):
        image = cv2.imread(os.path.join(SAVE_PATH, aug_images_list[idx]))
        mask = cv2.imread(os.path.join(SAVE_PATH, aug_images_list[idx + NUM_AUGMENTED_IMAGES]))
        dst = cv2.addWeighted(image, 0.7, mask, 0.4, 0)
        key = cv2.waitKey(round(1000 / FPS_RATE)) & 0xFF
        if (key == 27) or (key == 13):  # wait for ESC or Enter key to exit
            break
        cv2.imshow('Augmented image', dst)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()