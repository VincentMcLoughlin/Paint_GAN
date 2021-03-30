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

import numpy as np 
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
from keras.datasets import cifar10
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4 #0.4 works
session = tf.compat.v1.Session(config=config)

def scale_images(images, new_shape):
    images_list = list()

    for image in images:
        #resize with nearest neighbour interpolation
        new_image = resize(image, new_shape,0)
        #store
        images_list.append(new_image)

    return asarray(images_list)

def calculate_fid(model,images1,images2):

    #calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)

    #Calculate mean and covariance
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

    #calculate sum squared difference between means
    ssdiff = numpy.sum((mu1-mu2)**2.0)

    #sqrt of prod between cov. 
    covmean = sqrtmean(sigma1.dot(sigma2))

    if iscomplexobj(covmean):
        covmean = covmean.real

    #calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0*covmean)

    return fid

#Random image data
# images1 = randint(0,255, 10*32*32*3)
# images1 = images1.reshape((10,32,32,3))
# images2 = randint(0,255, 10*32*32*3)
# images2 = images2.reshape((10,32,32,3))

#Calculate between CIFAR10 train and test 
# load cifar10 images
(images1, _), (images2, _) = cifar10.load_data()

#Reduce training set to 10k random images to speed things up
shuffle(images1)
images1 = images1[:10000]


images1 = images1.astype('float32')
images2 = images2.astype('float32')

#resize images 
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))

images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

# load inception v3 model
model = InceptionV3()
#remove last layer and specify input
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

#fid between images1 and images1
fid = calculate_fid(model, images1, images1)
print('FID (same): %.3f' % fid)
# fid between images1 and images2
fid = calculate_fid(model, images1, images2)
print('FID (different): %.3f' % fid)

