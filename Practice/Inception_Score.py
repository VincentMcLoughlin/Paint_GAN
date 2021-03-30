#What is IS? Single float representing the quality of the GAN. Corresponds well to human 
#Interpretation 

#Measures
# 1. Images have variety
# 2. Image looks like something

#Based on inception network classifier
#We pass our generated image to our network and it tells us how likely it is 
#that we fall in a certain class. 

#We also sum a sample of all images in dataset (at least 50K images) to get 
#Marginal distribution (which gives us a distribution telling us how many images 
# are in each class)

#We can now measure quality and variety. We want each image to be distinct and 
# collectively to have variety. We can take a look at a collection of our generator
# images and actual images to see how the marginal distributions differ. The more they 
# differ, the higher the score, which is our inception score. 
# 
# We use KL divergence to calculate this. 
# 

import numpy as np 
import tensorflow as tf 
from tensorflow import keras
# calculate inception score with Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

def calculate_inception_score(images, n_split=10, eps=1E-16):

    # load inception v3 model
    model = InceptionV3()    

    #convert from uint8 to float32
    processed = images.astype('float32')

    # pre-process raw images for inception v3 model
    processed = preprocess_input(processed)

    #predict class probabilities from images. 
    yhat = model.predict(images)

    scores = list()
    #Must split conditional probabilites into groups 
    n_part = floor(images.shape[0]/n_split)

    for i in range(n_split):
        #Enumerate conditional probabilities over n_part images and Get p(y|x)
        ix_start, ix_end = i * n_part, (i+1) * n_part
        p_yx = yhat[ix_start:ix_end]

        p_y = expand_dims(p_yx.mean(axis=0),0)

        #K_L divergence for each image (log of prob * prob)
        kl_d = p_yx * (log(p_yx+eps) - log(p_y+eps))

        #sum over classes
        sum_kl_d = kl_d.sum(axis=1)

        #average over classes
        avg_kl_d = mean(sum_kl_d)

        #undo logs
        is_score = exp(avg_kl_d)

        scores.append(is_score)

    #Get mean and std dev
    is_avg, is_std = mean(scores), std(scores)
    
    return is_avg, is_std

# pretend to load images
images = ones((50, 299, 299, 3))
print('loaded', images.shape)
# calculate inception score
is_avg, is_std = calculate_inception_score(images)
print('score', is_avg, is_std)