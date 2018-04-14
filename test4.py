from keras.preprocessing.image import ImageDataGenerator
from data_processing import load_train_data

IMG_WIDTH = 512
IMG_HEIGHT = 512
number_of_samples = 1
train_data_dir = 'data/train/images'
test_data_dir = 'data/train/masks'


# Create two instances with the same arguments
datagen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=45,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     shear_range=0.2,
                     zoom_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='nearest')

image_datagen = ImageDataGenerator(**datagen_args)
mask_datagen = ImageDataGenerator(**datagen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
# images, masks = load_train_data()
# images = images.reshape(images.shape[0], images.shape[2], images.shape[3], images.shape[1])
# masks = masks.reshape(masks.shape[0], masks.shape[2], masks.shape[3], masks.shape[1])
# image_datagen.fit(images, augment=True, rounds=1, seed=seed)
# mask_datagen.fit(masks, augment=True, rounds=1, seed=seed)

image_generator = image_datagen.flow_from_directory(train_data_dir,
                                                    color_mode='grayscale',
                                                    target_size=(512, 512),
                                                    class_mode=None,
                                                    seed=seed)

mask_generator = mask_datagen.flow_from_directory(test_data_dir,
                                                  color_mode='grayscale',
                                                  target_size=(512, 512),
                                                  class_mode=None,
                                                  seed=seed)

# Combine generators into one which yields image and masks
def combineGenerator(gen1, gen2):
    while True:
        yield(gen1.next(), gen2.next())
train_generator = combineGenerator(image_generator, mask_generator)


model.fit_generator(train_generator,
                    steps_per_epoch=20,
                    epochs=5)
