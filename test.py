from keras.models import load_model
from utils.losses import dice_coef, jaccard_coef

model = load_model('model/hdf5 models/model.10-0.07-0.02-0.03-0.03.hdf5 models', custom_objects={'dice_coef': dice_coef, 'jaccard_coef': jaccard_coef})
model.summary()
print('dsada')