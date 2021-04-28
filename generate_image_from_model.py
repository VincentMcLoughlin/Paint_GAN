import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from numpy.random import randn
from numpy.random import randint
import cv2

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_shape = (64,64,3)
latent_dim = 100
n_samples = 1 #batch size
# g_model = tf.keras.models.load_model('128x128_augmented_models/generator_model_200.h5')
# filename = "Generated_Images_128x128"
g_model = tf.keras.models.load_model('64x64_augmented_models/generator_model_200.h5')
filename = "Generated_Images_64x64"

dpi_val = 96 #Monitor dependent, pixels per inch
actual_pic_size = img_shape[0]/dpi_val

fig = plt.figure(figsize=(10,10), dpi=dpi_val)
nplot = 11
for count in range(1,nplot):
    x_input = randn(latent_dim * n_samples) # generate input from the gaussian distributed latent space (just a random noise vector)                                            	
    x_input = x_input.reshape(n_samples, latent_dim) # reshape into a batch of inputs for the network
    X = g_model.predict(x_input) #Get generated images
    data = (X + 1) / 2.0
    ax = fig.add_subplot(1,nplot,count)
    index = np.random.randint(0,len(data))
    ax.imshow(data[index])
    plt.axis('off')

plt.gca().set_axis_off() #Remove white space and save, need all these lines
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
plt.show()