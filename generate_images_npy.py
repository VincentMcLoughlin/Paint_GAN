import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np 
from numpy import load
import tensorflow as tf 
from tensorflow import keras
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from numpy.random import randn
from numpy.random import randint
from numpy import save

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def get_generated_images(g_model, n_images):
    image_list = []
    latent_dim = 100
    n_samples = n_images
    total_input = randn(latent_dim * n_samples) # generate input from the gaussian distributed latent space (just a random noise vector)                                            	
    total_input = total_input.reshape(n_samples, latent_dim) # reshape into a batch of inputs for the network
    Tot = g_model.predict(total_input) #Get generated images 
    return Tot

num_images = 2500
g_model = tf.keras.models.load_model('128x128_augmented_models/generator_model_200.h5')
generated_data = get_generated_images(g_model, num_images)
print(generated_data.shape)
save('2.5K_128x128_generated_images.npy', generated_data)