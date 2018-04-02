from tf_unet import unet, util, image_util
import numpy as np

# Preparing data loading
data_provider = image_util.ImageDataProvider("data/train/*.tif")

# Setup & training
net = unet.Unet(layers=3,
                features_root=64,
                channels=1,
                n_class=2)
trainer = unet.Trainer(net,
                       optimizer="momentum",
                       opt_kwargs=dict(momentum=0.2))
path = trainer.train(data_provider,
                     "./unet_trained",
                     training_iters=10,
                     epochs=2,
                     dropout=0.25,
                     restore=False,
                     write_graph=False,
                     prediction_path=u'prediction')

# Verification
# prediction = net.predict(path, data)
# unet.error_rate(prediction, util.crop_to_shape(label, prediction.shape))
# img = util.combine_img_prediction(data, label, prediction)
# util.save_image(img, "prediction.jpg")

print('Done')