from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import Augmentor

FPS_RATE = 1
is_visualize = True
is_keras_augmentor = False
NUM_AUGMENTED_IMAGES = 10
IMG_HEIGHT = 512
IMG_WIDTH = 512
NUM_IMAGE = 85
SAVE_PATH = 'data/train/test augmentor/images/tumor/augmented examples'         # For keras augmentor
OUTPUT_PATH = 'augmented examples'                                              # For additional augmentor
SOURCE_PATH = 'data/train/test augmentor/images/tumor'                          # For additional augmentor
GROUND_TRUTH_PATH = 'data/train/test augmentor/masks/tumor'                     # For additional augmentor

# Remove old images
if is_keras_augmentor:
    file_list = os.listdir(SAVE_PATH)
    for file_name in file_list:
        os.remove(SAVE_PATH + '/' + file_name)
else:
    file_list = os.listdir(os.path.join(SOURCE_PATH, 'augmented examples/'))
    for file_name in file_list:
        os.remove(os.path.join(SOURCE_PATH, 'augmented examples/') + file_name)

# Augmentation
if is_keras_augmentor:
    datagen_args = dict(rotation_range=80,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.01,
                        zoom_range=0.25,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='nearest')

    datagen = ImageDataGenerator(**datagen_args)

    img = cv2.imread('data/train/images/tumor/' + str(NUM_IMAGE).zfill(3) + '_tumor.png', cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread('data/train/masks/tumor/' + str(NUM_IMAGE).zfill(3) + '_tumor_mask.png', cv2.IMREAD_GRAYSCALE)
    img = img.reshape([1, img.shape[0], img.shape[1], 1])
    mask = mask.reshape([1, mask.shape[0], mask.shape[1], 1])

    i = 0
    for batch in datagen.flow(img,
                              y=mask,
                              batch_size=1,
                              shuffle=True,
                              save_to_dir=SAVE_PATH,
                              save_prefix='augmented',
                              save_format='png'):
        i += 1
        if i > NUM_AUGMENTED_IMAGES-1:
            break  # otherwise the generator would loop indefinitely
else:
    pipeline = Augmentor.Pipeline(source_directory=SOURCE_PATH, output_directory=OUTPUT_PATH, save_format='png')
    pipeline.ground_truth(GROUND_TRUTH_PATH)
    pipeline.rotate(probability=0.3, max_left_rotation=25, max_right_rotation=25)           # Basic
    pipeline.rotate_random_90(0.3)                                                          # Basic
    pipeline.flip_random(probability=0.25)                                                  # Basic
    pipeline.zoom_random(probability=0.4, percentage_area=0.90)                             # Basic
    pipeline.random_distortion(probability=0.92, grid_width=8, grid_height=8, magnitude=6)  # Basic
    # pipeline.flip_left_right(probability=0.5)                                             # Additional
    # pipeline.flip_top_bottom(probability=0.5)                                             # Additional
    # pipeline.skew(probability=0.1, magnitude=0.5)                                         # Additional
    pipeline.sample(NUM_AUGMENTED_IMAGES)

# Visualize images
if is_visualize:
    if is_keras_augmentor:
        SAVE_PATH = SAVE_PATH
        for _, file in enumerate(os.listdir(SAVE_PATH), start=1):
            image = cv2.imread(os.path.join(SAVE_PATH, file))
            key = cv2.waitKey(round(1000 / FPS_RATE)) & 0xFF
            if (key == 27) or (key == 13):  # wait for ESC or Enter key to exit
                break
            cv2.imshow('Augmented image', image)
        cv2.waitKey(0)
    else:
        SAVE_PATH = os.path.join(SOURCE_PATH,OUTPUT_PATH)
        aug_images_list = os.listdir(SAVE_PATH)
        for idx in range(NUM_AUGMENTED_IMAGES):
            image = cv2.imread(os.path.join(SAVE_PATH, aug_images_list[idx]))
            mask = cv2.imread(os.path.join(SAVE_PATH, aug_images_list[idx+NUM_AUGMENTED_IMAGES]))
            dst = cv2.addWeighted(image, 0.7, mask, 0.4, 0)
            key = cv2.waitKey(round(1000 / FPS_RATE)) & 0xFF
            if (key == 27) or (key == 13):  # wait for ESC or Enter key to exit
                break
            cv2.imshow('Augmented image', dst)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
