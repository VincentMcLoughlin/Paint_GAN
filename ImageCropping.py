#!/usr/bin/env python
#https://stackoverflow.com/questions/9103257/resize-image-maintaining-aspect-ratio-and-making-portrait-and-landscape-images-e

from PIL import Image, ImageChops, ImageOps

def makeThumb(f_in, f_out, size=(128,128), pad=False):

    image = Image.open(f_in)
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    if pad:
        thumb = image.crop( (0, 0, size[0], size[1]) )

        offset_x = int(max( (size[0] - image_size[0]) / 2, 0 ))
        offset_y = int(max( (size[1] - image_size[1]) / 2, 0 ))   

        thumb = ImageChops.offset(thumb, offset_x, offset_y)

    else:
        print(Image.ANTIALIAS)
        thumb = ImageOps.fit(image, size, Image.ANTIALIAS, 0,(0.5, 0.5))

    thumb.save(f_out)


source = "wikiart/wikiart/Impressionism/abdullah-suriosubroto_bamboo-forest.jpg"

makeThumb(source, "image_padded.JPG", pad=True)
makeThumb(source, "image_centerCropped.JPG", pad=False)