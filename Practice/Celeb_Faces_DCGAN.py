import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import keras
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array

# See https://fairyonice.github.io/My-first-GAN-using-CelebA-data.html
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

dir_data = "processed-celeba-small/processed_celeba_small/celeba/New Folder With Items"
#"Practice/processed-celeba-small/processed_celeba_small/celeba/161979.jpg"
#"Practice/processed-celeba-small/processed_celeba_small/celeba/New Folder With Items/000001.jpg"
#32,600 images total, 64x64x3
#57331 in 
nTrain = 16600
nTest = 8000
nm_imgs = np.sort(os.listdir(dir_data))
nm_imgs_train = nm_imgs[:nTrain]
nm_imgs_test = nm_imgs[nTrain:nTrain+nTest]

img_shape = (32,32,3)

def get_np_data(nm_imgs_train):
    x_train = []

    for i, my_id in enumerate(nm_imgs_train):
        image = load_img(dir_data + "/" + my_id, target_size = img_shape[:2])
        image = img_to_array(image)/255.0
        x_train.append(image)
    x_train = np.array(x_train)
    return (x_train)

X_train = get_np_data(nm_imgs_train)
print("X_train.shape = {}".format(X_train.shape))

X_test  = get_np_data(nm_imgs_test)
print("X_test.shape = {}".format(X_test.shape))

fig = plt.figure(figsize=(30,10))
nplot = 7
for count in range(1,nplot):
    ax = fig.add_subplot(1,nplot,count)
    ax.imshow(X_train[count])
plt.show()

learning_rate = 1e-4
generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

def make_generator_model():
    model = tf.keras.Sequential()

    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape(4,4,1024))
    assert model.output_shape == (None, 4,4,1024)

    model.add(layers.Conv2DTranspose(512, (5,5), strides=(2,2), padding="same", use_bias=False))
    assert model.output_shape == (None, 8,8,512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding="same", use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding="same", use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5,5), padding="same", use_bias=False, activation="sigmoid")) #Why not tanh? May be worth trying. Also 128 to 3 seems a little abrupt
    assert model.output_shape == (None, 64, 64, 3)

    return model 

def make_discriminator_model():
    model = tf.layers.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding="same", input_shape=[64,64,3]))
    model.add(layers.LeakyReLU)
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding="same"))
    model.add(layers.LeakyReLU)
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

@tf.function 
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss 
    return total_loss 

@tf.function 
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(tf.ones_like(fake_output), fake_output))


def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    plt.close(fig)


