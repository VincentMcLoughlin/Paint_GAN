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
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_shape = (32,32,3)

def get_np_data(nm_imgs_train):
    x_train = []

    for i, my_id in enumerate(nm_imgs_train):
        image = load_img(dir_data + "/" + my_id, target_size = img_shape[:2])
        image = img_to_array(image)/255.0
        x_train.append(image)
    x_train = np.array(x_train)
    return (x_train)

learning_rate = 1e-4
generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256, use_bias=False, input_shape=(100,))) #Input is 7*7*256, 256 is batch size, 7 is image size (grows to 28). 100 input shape is noise tensor size
    model.add(layers.BatchNormalization()) #Always batch normalize except for intermediate layers at gen output and disc input. 
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4, 4, 256))) #Reshape to fit. 
    assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))     
    assert model.output_shape == (None, 4, 4, 128) #Decrease third channel
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))    
    assert model.output_shape == (None, 8, 8, 64) #Decrease again, increase image size
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))    
    assert model.output_shape == (None, 16, 16, 32) #Decrease again, increase image size
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))    
    assert model.output_shape == (None, 32, 32, 3) #Get final output

    return model 

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', 
                                     input_shape=[32, 32, 3])) #filter size of 64 (filters  = num of output filters in convolution, e.g output size in neurons)
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) 

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')) 
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1)) #output decision, fake/not

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

dir_data = "processed-celeba-small/processed_celeba_small/celeba/New Folder With Items"
#"Practice/processed-celeba-small/processed_celeba_small/celeba/161979.jpg"
#"Practice/processed-celeba-small/processed_celeba_small/celeba/New Folder With Items/000001.jpg"
#32,600 images total, 64x64x3
#57331 in 
nTrain = 16600
nTest = 8000
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

# fig = plt.figure(figsize=(30,10))
# nplot = 7
# for count in range(1,nplot):
#     ax = fig.add_subplot(1,nplot,count)
#     ax.imshow(X_train[count])
# plt.show()

generator = make_generator_model() 
discriminator = make_discriminator_model()
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer, seed)

