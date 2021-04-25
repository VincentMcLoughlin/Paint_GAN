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

#Need three classes to achieve this progressive growth 
#1. Weighted Sum Layer to control sum of old and new layers 
#2. Minibatch Stddev Used to sumarize statistics for a batch of images in the generator 
# Pixel Normalization Used to normalize activation maps in the generator model

class WeightedSum(Add):
    #Adds activations when we are transitioning from one image size to another
    #Alpha scales from 0 at the start to 1 at the end so we start with all the weight 
    #From the old smaller layer and finish with all the weight from the new larger layer
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

        #output weighted sum of inputs 
        def _merge_function(self, inputs):
            assert(len(inputs)==2)
            #((1-a)*input1) * (a*input2)
            output = ((1.0-self.alpha)*inputs[0]) + (self.alpha * inputs[1])

#The minibatchStdev class calculates statistics about the images produced by the generator
#and is passed to the discriminator to help it learn what statisitcs real images should have
#This then encourages the generator to produce images with realistic batch statistics

#We calculate the std dev for each pixel value in the activation maps across the batch, 
#Calculating the average of this value, and then create a new activation map with one channel
#That is appended to the list of input activation maps
class MinibatchStdev(Layer):

    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    #Perform the operation
    def call(self, inputs):
        mean = backend.mean(inputs, axis=0, keepdims=True) #Mean for each pixel across channels

        squ_diffs = backend.square(inputs-mean) #Sq diff between pixel value and mean

        mean_sq_diffs = backend.mean(squ_diffs, axis=0, keepdims=True) #Avg sq diff

        mean_sq_diff += 1e-8 #Add small value to avoid explosion when calculating stddev

        stdev = backend.sqrt(mean_sq_diff) #Sq root of variance

        mean_pix = backend.mean(stdev, keepdims=True) #Mean std dev across each pixel coordinate

        shape = backend.shape(inputs) #Scale this up to feature map size
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1)) 

        combined = backend.concatenate([inputs, output], axis=-1) #concatenate with output

        return combined 

    def compute_output_shape(self, input_shape):

        input = list(input_shape) #Create copy of input shape as list
        input_shape[-1] += 1 #add one to channel dimension (assuming last)

        return tuple(input_shape) #convert list to a tuple

# Gen and Disc models don't use batch normalization like other GAN models. Instead each pixel
# in the activation maps is normalizaed to unit length. Called pixelwise feature vector 
# normalization. Used only in generator (not discriminator). Used after each conv layer
# but before activation.
class PixelNormalization(Layer):

    #Initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    def call(self, inputs):

        values = inputs**2.0 #Square pixel values

        mean_values = backend.mean(values, axis=-1, keepdims=True) #Get mean pixel val

        mean_values += 1.0e-8 #Ensure mean is non-zero

        L2 = backend.sqrt(mean_values) #Calculate the sqrt of mean square value (L2 Norm)

        normalized = inputs/L2 #Normalize values by L2 norm
        return nromalized 

    def compute_output_shape(sef, input_shape):
        return input_shape        


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

#Discriminator model
# We start off receiving a 4x4 colour image and predict whether it is real or fake. 
# First hidden layer is a 1x1 convolution layer. FOr the output block we have a 
# minibatchstdev, 3x3 and 4x4 convolution layers and a FC layer that outputs a prediction
#Leaky Relu used except for output layers which uses a linear activation function. 

# Model trained for a normal interval then the model undergoes growth to 8x8. 
# Two 3x3 conv blocks are added and an average pooling downsample layer is added. 
# The input image passes through the new block with a new 1x1 convolution hidden layer
# Image also downsampled and passed through the old 1x1 conv hidden layer. The original 
# 1x1 and new conv layer are combined using a WeightedSum layer. 

# After an interval of training where we transition weightedSums alpha parameter from 0 (all old)
# to 1 (all new) another training phase is run to tune the new model with the old layer and 
# the pathway is removed. 

# We repeat this process until the desired image size is obtained. 

#We will use wasserstein loss which is suggested in the original paper. 
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true*y_pred)

def add_discriminator_block(old_model, n_input_layers=3):
    init = RandomNormal(stddev=0.02) #intiialize weights

    const = max_norm(1.0) #Weight constraints

    in_shape = list(old_model.input_shape) #Get existing model shape

    #define new shape as double the size
    input_shape = (in_shape[-2].value*2, in_shape[-2].value*2, in_shape[-1].value) 
    in_image = Input(shape=input_shape)

    #define new input processing layer
    d = Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d=LeakyReLU(alpha=0.2)(d)

    #define new block
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D()(d)
    block_new = d

    # Skip input, 1x1 and activation for old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    #define straight through model
    model1 = Model(in_image, d)

    #compile model
    model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    #downsample new larger image 
    downsample = AveragePoolin2D()(in_image)

    #Connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)

    #Fade in output of old model input layer with new input
    d = WeightedSum()([block_old, block_new])

    #Skip input 1x1 and activations for old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)

    #Straight through
    model2 = Model(in_image, d)

    #compile model
    model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    return [model1, model2]

#define discriminator models for each resolution
def define_discriminator(n_blocks, input_shape=(4,4,3)):

    #Weight initialization
    init = RandomNormal(stddev=0.02)

    const = max_norm(1.0)
    model_list = list()

    #base model input
    in_image = Input(shape=input_shape)

    #Conv 1x1 
    d = COnv2D(128, (1,1), padding="same", kernel_initializer=init, kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    #COnv 3x3 (output block)
    d = MinibatchStdev()(d)
    d = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)

    #conv 4x4
    d = COnv2D(128, (4,4), padding="same", kernel_initializer=init, kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)

    #dense output layer
    d = Flatten()(d)
    out_class = Dense(1)(d)

    #define model
    model = Model(in_image, out_class)

    #compile model
    model.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

    #Store model
    model_list.append([model, model])

    #Create submodels
    for i in range(1, n_blocks):
        #get prior model without fade on
        old_model = model_list[i-1][0]
        #create new model for next resolution
        models = add_discriminator_block(old_model)
        #store model 
        model_list.append(models)
    return model_list 

# Generator model works the same way. First generate a 4x4 and then go progressively larger
# Defined in a similar way to the discriminator models. Define a 4x4 base model and then 
# define growth versions of the model for producing larger outputs.
# 
# Main difference is output of model is output of weightedSum layer. The growth phase 
# Version of the model involves first adding a nearest nieghbour upsampling layer. This
# is connected to the new block the new and old output layers and these old and new layers 
# are then combined using the weightedSum output layer.

def add_generator_block(old_model):
    #weight init
    init = RandomNormal(stddev=0.02)

    #weight constraint 
    const = max_norm(1.0)

    block_end = old_model.layers[-2].output

    #upsample and define new block
    upsampling = UpSampling2D()(block_end)
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)

    #add a new output layer
    out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)

    #define model
    model1 = Model(old_model.input, out_image)

    #get output layer from old model
    out_old = old_model.layers[-1]

    #connect upsampling layer to old output layer 
    out_image2 = out_old(upsampling)

    #Sum old and new models
    merged = WeightedSum()[out_image2, out_image]

    #define model
    model2 = Model(old_model.input, merged)

    return [model1, model2]

#define generator models
def define_generator(latent_dim, n_blocks, in_dim=4):

    #weight init
    init = RandomNormal(stddev=0.02)

    const = max_norm(1.0)
    model_list = list()

    #base model latnet input 
    in_latent = Input(shape=(latent_dim,))

    #linear scale up to activation maps
    g = Dense(128 * in_dim * in_dim, kernel_initilaizer=init, kernel_constraint=const)(in_latent)
    g = Reshape((in_dim, in_dim, 128))(g)

    #conv 4x4, input block
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNoramlization()(g)
    g = LeakyReLU(alpha=0.2)(g)

    #conv 3x3 
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)

    #conv 1x1 output block
    out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)

    #define model
    model = Model(in_latent, out_image)

    #store model
    model_list.append([model,model])

    #create submodels
    for i in range(1, n_blocks):
        old_model = model_list[i-1][0]

        #create model for next resolution
        models = add_generator_block(old_model)

        #store model
        model_list.append(models)

    return model_list

# We don't train the generator models directly. Instead we use the discriminators wass. loss
# We pair each generator model up with a discrminator model for the same size and get them 
# to train each other. This is achieved by creating a new model for each pair that stacks
# the gen onto the disc so the output goes directly to the discriminator. 
# We then train the composite model which trains the generator. Discriminator is trained 
# separately

def define_composite(discriminators, generators):
    model_list = list()

    #Create composite models
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]

        #straight through model
        d_models[0].trainable = False

        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        #fade in model
        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss, optimizer=Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        model_list.append([model1, model2])

    return model_list

# Training has two stages, the fade in from lower to higher resolution, and the normal phase 
# that involves the fine-tuning of models at the higher resolution.
# The update fadein function takes a list of models, your gen, disc, and comp and sets the
# alpha value for each attribute based on the current training step number
def update_fadein(models, step, n_steps):
    #calculate current alpha
    alpha = step/float(n_steps-1)

    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                backend.set_value(layer.alpha, alpha)

def scale_dataset(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25):
	# devise name
	gen_shape = g_model.output_shape
	name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
	# generate images
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# normalize pixel values to the range [0,1]
	X = (X - X.min()) / (X.max() - X.min())
	# plot real images
	square = int(sqrt(n_samples))
	for i in range(n_samples):
		pyplot.subplot(square, square, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X[i])
	# save plot to file
	filename1 = 'plot_%s.png' % (name)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%s.h5' % (name)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_steps):
		# update alpha for all WeightedSum layers when fading in new blocks
		if fadein:
			update_fadein([g_model, d_model, gan_model], i, n_steps)
		# prepare real and fake samples
		X_real, y_real = generate_real_samples(dataset, half_batch)
		X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model
		d_loss1 = d_model.train_on_batch(X_real, y_real)
		d_loss2 = d_model.train_on_batch(X_fake, y_fake)
		# update the generator via the discriminator's error
		z_input = generate_latent_points(latent_dim, n_batch)
		y_real2 = ones((n_batch, 1))
		g_loss = gan_model.train_on_batch(z_input, y_real2)
		# summarize loss on this batch
		print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))
    
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
	# fit the baseline model
	g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
	# scale dataset to appropriate size
	gen_shape = g_normal.output_shape
	scaled_data = scale_dataset(dataset, gen_shape[1:])
	print('Scaled Data', scaled_data.shape)
	# train normal or straight-through models
	train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
	summarize_performance('tuned', g_normal, latent_dim)
	# process each level of growth
	for i in range(1, len(g_models)):
		# retrieve models for this level of growth
		[g_normal, g_fadein] = g_models[i]
		[d_normal, d_fadein] = d_models[i]
		[gan_normal, gan_fadein] = gan_models[i]
		# scale dataset to appropriate size
		gen_shape = g_normal.output_shape
		scaled_data = scale_dataset(dataset, gen_shape[1:])
		print('Scaled Data', scaled_data.shape)
		# train fade-in models for next level of growth
		train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
		summarize_performance('faded', g_fadein, latent_dim)
		# train normal or straight-through models
		train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
		summarize_performance('tuned', g_normal, latent_dim)

# number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
n_blocks = 6
# size of the latent space
latent_dim = 100
# define models
d_models = define_discriminator(n_blocks)
# define models
g_models = define_generator(latent_dim, n_blocks)
# define composite models
gan_models = define_composite(d_models, g_models)
# load image data
dataset = load('impressionism_128x128.npy')
#X_train = get_np_data(nm_imgs_train)
print('Loaded', dataset.shape)
# train model
n_batch = [16, 16, 16, 8, 4, 4]
# 10 epochs == 500K images per training phase
n_epochs = [5, 8, 8, 10, 10, 10]
train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)