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
import keras
import matplotlib.pyplot as plt
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
from tensorflow.keras import layers
from matplotlib import pyplot
from numpy import load
from tensorflow.keras.utils import plot_model
import time
from IPython import display
import glob
import imageio


physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

img_shape = (128,128,3)

def get_np_data(nm_imgs):
    x_train = []

    for file_name in nm_imgs:
        image = load_img(dir_data + "/" + file_name, target_size = img_shape[:2])
        image = (img_to_array(image) - 127.5)/127.5 #Gives -1 to 1 range
        x_train.append(image)
    x_train = np.array(x_train)
    x_train = x_train.astype('float32')
    return (x_train)

# select real samples
def generate_real_samples(dataset, n_samples):	
	ix = randint(0, dataset.shape[0], n_samples) #Pick random indices
	X = dataset[ix]	#Get samples
	y = ones((n_samples, 1)) #Assign real class labels (1)
	return X, y

def generate_latent_points(latent_dim, n_samples):	
	x_input = randn(latent_dim * n_samples) # generate input from the gaussian distributed latent space (just a random noise vector)                                            	
	x_input = x_input.reshape(n_samples, latent_dim) # reshape into a batch of inputs for the network
	return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)	
    X = g_model.predict(x_input) #Get generated images
    y = zeros((n_samples, 1)) # create 'fake' class labels (0)
    return X, y

def define_generator(latent_dim):
    model = Sequential()

    n_nodes = 256*4*4 #Good number of nodes to start off with, start off with 4x4 image and enough nodes in the dense layer to approximate picture
                      #Value needs to be large enough to model a variety of different features from the latent input space
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(layers.BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Reshape((4,4,256)))
    assert model.output_shape == (None, 4, 4, 256) 

    #Upsample to 8x8
    #Seems to be max our memory can take. Generally want to pick a kernel and stride as multiples of each other, avoids checkerboard output
    #The (2,2) stride results in doubling the height and width
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")) 
    print(model.output_shape)
    assert model.output_shape == (None, 8, 8, 128) #Decrease third channel
    #model.add(layers.BatchNormalization())
    model.add(LeakyReLU(alpha=0.2)) #This slope is best practice according to the guide? Not sure where they saw this, defaults to 0.3 according to tf website

    #Upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same"))
    print(model.output_shape)
    assert model.output_shape == (None, 16, 16, 128) #Decrease again, increase image size
    #model.add(layers.BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    #Upsample to 32x32
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")) 
    assert model.output_shape == (None, 32, 32, 128) #Decrease again, increase image size
    #model.add(layers.BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    #Upsample to 64x64
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")) 
    assert model.output_shape == (None, 64, 64, 128) #Decrease again, increase image size
    #model.add(layers.BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    #Upsample to 128x128
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding="same")) 
    assert model.output_shape == (None, 128, 128, 128) #Decrease again, increase image size
    #model.add(layers.BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    #Output
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))

    opt = Adam(lr=learning_rate, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_discriminator(in_shape=img_shape): #Do we really need this many layers in the discriminator?

    model = Sequential()

    #Normal 
    model.add(Conv2D(64, (3,3), padding="same", input_shape=in_shape))
    model.add(layers.LeakyReLU(alpha=0.2))

    # #Downsample 64x64
    # model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) 
    # #model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU(alpha=0.2))

    #Downsample 32x32
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) 
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    #Downsample to 16x16
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) 
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    #Downsample to 8x8
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')) 
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))    

    #Downsample to 4x4
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')) 
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))    

    model.add(layers.Flatten())
    model.add(Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid')) #output decision, fake/not, should we use relu here? They don't in tf example. Lets try sigmoid first and then see

    opt = Adam(lr=learning_rate, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model 

#Create a stacked gan to make things a bit easier to work with
#Will have just one big network that takes in a noise vector from the latent space and outputs a binary real/fake
def define_gan(g_model, d_model):
	d_model.trainable = False #Making weights not trainable is a good trick. The discriminator will be updated 
    #When we place calls to the train_on_batch is placed. We call train_on_batch on the discriminator 
    #separately 

	# connect them
	model = Sequential()	
	model.add(g_model)	
	model.add(d_model)

	# compile model
	opt = Adam(lr=learning_rate, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# create and save a plot of generated images
def save_plot(examples, epoch, n=7):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i])
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
 
# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch+1)
	g_model.save(filename)

def save_losses(losses):
    losses_file = open("128x128_losses.csv", "w")    
    np.savetxt(losses_file, losses, delimiter=",")        
    losses_file.close()

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    losses = [] #Epoch Batch d_loss1 d_loss2 g_loss

    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real) #training twice per batch is best practice (once with real once with fake)			

            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch) #Get fake samples
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake) #training twice per batch is best practice 
            
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = ones((n_batch, 1)) #Label as 1 because we want the discriminator to think they're real
            g_loss = gan_model.train_on_batch(X_gan, y_gan)  # update the generator via the discriminator's error, should be between 1.5-2+

            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

            losses.append([i,j,d_loss1,d_loss2,g_loss])

                # evaluate the model performance, sometimes
        if (i+1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)

        # losses[i,0] = i
        # losses[i,1] = d_loss1
        # losses[i,2] = d_loss2        
        # losses[i,3] = g_loss        
    losses = np.array(losses)
    save_losses(losses)

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

#dir_data = "wikiart/wikiart/Impressionism"
#dir_data = "Practice/processed-celeba-small/processed_celeba_small/celeba/New Folder With Items"
#image_names = np.sort(os.listdir(dir_data))
#nTest = 8000
#nTrain = len(image_names) - nTest

nTrain = 12000
nTest = 1000

BUFFER_SIZE = nTrain
BATCH_SIZE = 128
EPOCHS = 200
latent_dim = 100

learning_rate = 0.0002

generator = define_generator(latent_dim) 
#plot_model(generator, to_file='generator_plot.png', show_shapes=True, show_layer_names=True) #Useful for reports

discriminator = define_discriminator()
#plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True) #useful for reports

gan_model = define_gan(generator,discriminator)
#plot_model(gan_model, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

num_examples_to_generate = 16
#seed = tf.random.normal([num_examples_to_generate, latent_dim])
data = load('impressionism_128x128.npy')
#X_train = get_np_data(nm_imgs_train)
X_train = data[:nTrain]
print("X_train.shape = {}".format(X_train.shape))

#X_test  = get_np_data(nm_imgs_test)
X_test = data[nTrain:nTrain+nTest]
print("X_test.shape = {}".format(X_test.shape))

fig = plt.figure(figsize=(30,10))
nplot = 7
for count in range(1,nplot):
    ax = fig.add_subplot(1,nplot,count)
    ax.imshow(X_train[count])
plt.show()

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)

#train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#train(train_dataset, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer, seed)
train(generator, discriminator, gan_model, X_train, latent_dim, EPOCHS)