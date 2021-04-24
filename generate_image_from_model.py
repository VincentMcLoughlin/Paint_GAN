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
g_model = tf.keras.models.load_model('generator_model_200.h5')

dpi_val = 96 #Monitor dependent, pixels per inch
actual_pic_size = img_shape[0]/dpi_val

while True:
    x_input = randn(latent_dim * n_samples) # generate input from the gaussian distributed latent space (just a random noise vector)                                            	
    x_input = x_input.reshape(n_samples, latent_dim) # reshape into a batch of inputs for the network
    X = g_model.predict(x_input) #Get generated images
    data = (X + 1) / 2.0
    plt.figure(figsize=(2, 2), dpi=dpi_val)
    plt.imshow(data[0])
    plt.show()
    filename = 'tmp_pic.png'
    plt.savefig(filename)