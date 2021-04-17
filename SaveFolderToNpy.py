from numpy import asarray
from numpy import save
import os
import numpy as np
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_shape = (64,64,3)

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal"),
])

def prepare(ds):
    
    ds = ds.map(lambda x,y: (data_augmentation(x, training=True), y), num_parallel_calls=8)
    return ds

def get_np_data(nm_imgs):
    x_train = []

    for file_name in nm_imgs:
        image = load_img(dir_data + "/" + file_name, target_size = img_shape[:2])
        image = (img_to_array(image) - 127.5)/127.5 #Gives -1 to 1 range
        x_train.append(image)
    x_train = np.array(x_train)
    return (x_train)

dir_data = "wikiart/wikiart/Impressionism"
image_names = np.sort(os.listdir(dir_data))

# define data
data = get_np_data(image_names)
#train_dataset = tf.data.Dataset.from_tensor_slices(data)
augmented_data = tf.image.flip_left_right(data)
total = np.concatenate((data, augmented_data))
# save to npy file
save('impressionism_64x64_augmented.npy', total)

print(data.shape)
print(augmented_data.shape)
print(total.shape)