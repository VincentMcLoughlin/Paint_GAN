import numpy as np
from numpy import save
from numpy import load
import os 
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

output = '128x128_center_cropped_augmented.npy'
images = '128x128_center_cropped.npy'
images = np.load(images).astype('float32')
print(images.shape)
flipped_images = tf.image.flip_left_right(images)
total_images = np.concatenate((images, flipped_images))
print(total_images.shape)
save(output , total_images)