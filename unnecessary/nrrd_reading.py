import nrrd
import matplotlib.pyplot as plt
import cv2
import time

is_image_vis = 0
is_video_vis = 0

# Read the data
filename = 'data/LV Catheter 03.nrrd'
start = time.time()
nrrd_array, options = nrrd.read(filename)
end = time.time()
processing_time = end - start
dataset_size = options['sizes']
print('Dataset size: %s' % str(dataset_size))
print('Processing time: %2f' % processing_time)

# Image visualization
if is_image_vis == 1:
    num_of_slice = 30
    img = nrrd_array[:, :, num_of_slice, 0].T
    plt.imshow(img, interpolation='bicubic', cmap='gray', origin='upper')
    plt.show()

# Video visualization
if is_video_vis == 1:
    for num_of_slice in range(0, 208, 1):
        img = nrrd_array[:, :, num_of_slice, 0].T
        img = plt.imshow(img, interpolation='bicubic', cmap='gray', origin='upper')
        plt.pause(.01)
        plt.xticks([]), plt.yticks([])
        plt.title(str(num_of_slice) + ' slice')
        plt.draw()

# OpenCV visualization
is_openCV = 1
num_of_timeframe = 1
fps_rate = 30
if is_openCV == 1:
    for num_of_slice in range(1, 208, 1):
        img = nrrd_array[:, :, num_of_slice, num_of_timeframe - 1].T
        cv2.namedWindow('%d timeframe' % num_of_timeframe, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('%d timeframe' % num_of_timeframe, 500, 500)
        # font = cv2.FONT_HERSHEY_TRIPLEX
        # cv2.putText(img, str(nSlice) + ' slice', (1, 10), font, 0.5, 255, 1)
        cv2.imshow('%d timeframe' % num_of_timeframe, img)
        k = cv2.waitKey(round(1000/fps_rate)) # >0 - play, 0 - play with the pressing any key
        if k == 27:  # wait for ESC key to exit
            break
        elif k == ord('s'):  # wait for 's' key to save and exit
            cv2.imwrite('data/%d slice gray.png' % num_of_slice, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.destroyWindow('%d Display window' % num_of_slice)