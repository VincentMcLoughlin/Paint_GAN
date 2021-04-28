#Measures distance between feature vectors of real images and feature vectors of fake images
#Therefore lower is better (with 0 being perfectly identical)

#FID effectively detects Gaussian noise, Gaussian blur, Implanted block rectangles, swirled images, 
#salt and pepper noise, and cross contamination from multiple images fusing. 

#Better than inception score as it is robust to noise, image distortions, and pertrubations
#Once again use inception model to extract feature maps. 

#To calculate 
#Load a pretrained Inception v3 model 
#Remove output layer, and take output as the output of the activations from the last layer 
#output has 2048 activation features so each image is predicted as 2048 activation features. 

#We then get 2048 features for real and generated and compare them 
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

def scale_images(images, new_shape):
    images_list = list()

    for image in images:
        #resize with nearest neighbour interpolation
        new_image = resize(image, new_shape,0)
        #store
        images_list.append(new_image)

    return asarray(images_list)

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	
    # calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	
    # calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	
    # calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	
    # check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

#Calculate between real data and generated
# load images
num_images = 2500
# real_data = load('impressionism_128x128_augmented.npy')
# generated_data = load("10K_128x128_generated_images.npy") #FID (different): 2.524

# real_data = load('impressionism_128x128_augmented.npy') #FID (different): 2.486
# generated_data = load("2.5K_128x128_generated_images.npy")

# real_data = load('impressionism_64x64_augmented.npy')
# generated_data = load("10K_64x64_generated_images.npy") #FID: 1.198 

real_data = load('impressionism_64x64_augmented.npy')
generated_data = load("2.5K_64x64_generated_images.npy")

#Reduce training set to 10k random images to speed things up
shuffle(real_data)
real_data = real_data[:num_images]

real_data = real_data.astype('float32')
generated_data = generated_data.astype('float32')

#resize images 
real_data = scale_images(real_data, (299,299,3)) #Required by Inception model
generated_data = scale_images(generated_data, (299,299,3))

real_data = preprocess_input(real_data)
generated_data = preprocess_input(generated_data)

# load inception v3 model
model = InceptionV3()
#remove last layer and specify input
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# fid between real_data and generated_data
fid = calculate_fid(model, real_data, generated_data)
print('FID : %.3f' % fid)

