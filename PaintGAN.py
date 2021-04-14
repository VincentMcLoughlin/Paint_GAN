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

def get_np_data(nm_imgs_train):
    x_train = []

    for i, my_id in enumerate(nm_imgs_train):
        image = load_img(dir_data + "/" + my_id, target_size = img_shape[:2])
        #image = img_to_array(image)/255.0 #This normalization is likely wrong, too compressive, on 0-1 range
        image = (img_to_array(image) - 127.5)/127.5 #Gives -1 to 1 range
        x_train.append(image)
    x_train = np.array(x_train)
    return (x_train)


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
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer, seed):

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,epoch + 1,seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print ('Time for epoch {} is {} sec. Gen loss:{} Disc loss:{} '.format(epoch + 1, time.time()-start, gen_loss, disc_loss))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    print(predictions.shape)    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)                    
        plt.imshow((predictions[i, :, :, :]+1)/2) #Scale images from [-1,1] to [0,1]         
        plt.axis('off')
    #print(predictions[i,:,:,:] * 127.5 + 127.5)
    plt.savefig('output/image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()
    plt.close(fig)

dir_data = "wikiart/wikiart/Impressionism"

nTest = 8000
nTrain = 57331-nTest
BUFFER_SIZE = nTrain
BATCH_SIZE = 256 #originally 256
EPOCHS = 200
noise_dim = 100
nm_imgs = np.sort(os.listdir(dir_data))
nm_imgs_train = nm_imgs[:nTrain]
nm_imgs_test = nm_imgs[nTrain:nTrain+nTest]
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

X_train = get_np_data(nm_imgs_train)
print("X_train.shape = {}".format(X_train.shape))

X_test  = get_np_data(nm_imgs_test)
print("X_test.shape = {}".format(X_test.shape))


generator = make_generator_model() 
discriminator = make_discriminator_model()
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer, seed)