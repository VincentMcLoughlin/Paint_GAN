#!/usr/bin/env python
#https://stackoverflow.com/questions/9103257/resize-image-maintaining-aspect-ratio-and-making-portrait-and-landscape-images-e

from PIL import Image, ImageChops, ImageOps
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from numpy import save
import glob
import tensorflow as tf
resolution = 128

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def makeThumb(f_in, f_out, size=(resolution,resolution), pad=False):

    images = []
    i = 0
    for f in glob.iglob(f_in):    

        image = Image.open(f)
        image.thumbnail(size, Image.ANTIALIAS)
        image_size = image.size

        if pad:
            thumb = image.crop( (0, 0, size[0], size[1]) )

            offset_x = int(max( (size[0] - image_size[0]) / 2, 0 ))
            offset_y = int(max( (size[1] - image_size[1]) / 2, 0 ))

            thumb = ImageChops.offset(thumb, offset_x, offset_y)

        else:            
            thumb = ImageOps.fit(image, size, Image.ANTIALIAS, 0.1,(0.5, 0.5))

        thumb = )(np.asarray(thumb) - 127.5)/127.5).astype('float32')
        images.append(thumb)
        print(i)
        i += 1

    # flipped_images = tf.image.flip_left_right(images)
    # total_images = np.concatenate((images, flipped_images))
    save(f_out, images)
    # save(f_out, total_images)
    


source = "wikiart/wikiart/Impressionism/*.jpg"
output = '128x128_center_cropped.npy'
#source = "tmp_folder/*.jpg"
#dir_data = "wikiart/wikiart/Impressionism"
#makeThumb(source, "image_padded.JPG", pad=True)
makeThumb(source, output, pad=False)

#print(os.listdir(dir_data))