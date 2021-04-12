import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
import tensorflow as tf
from keras.datasets.cifar10 import load_data
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
img_shape = (64,64,3)

def define_generator():
    model = Sequential()

    n_nodes = 256*4*4 #Good number of nodes to start off with, start off with 4x4 image and enough nodes in the dense layer to approximate picture
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(LeakyRelU())
    model.add(Reshape(4,4,256))

    #Upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")) #Seems to be max our memory can take
    model.add(layers.BatchNormalization())
    model.add(LeakyRelU())

    #Upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")) 
    model.add(layers.BatchNormalization())
    model.add(LeakyRelU())

    #Upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")) 
    model.add(layers.BatchNormalization())
    model.add(LeakyRelU())

    #Upsample to 64x64
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")) 
    model.add(layers.BatchNormalization())
    model.add(LeakyRelU())

    #Output
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    return model

def define_discriminator(in_shape=img_shape): #Do we really need this many layers in the discriminator?

    model = Sequential()

    #Normal 
    model.add(Conv2D(64, (3,3), padding="same", input_shape=in_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Downsample 32x32
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) 
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Downsample to 16x16
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) 
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Downsample to 8x8
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) 
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    #Downsample to 4x4
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')) 
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='relu')) #output decision, fake/not, should we use relu here? They don't in tf example

    return model 

@tf.function 
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss 
    return total_loss 

@tf.function 
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    