import numpy as np
from matplotlib import pyplot as plt
from numpy import load


data = load('impressionism_128x128_augmented.npy')
filename = "Real Images 128x128"
#data = load('impressionism_64x64_augmented.npy')
#filename = "Real Images 64x64"
img_shape = (128,128,3)

fig = plt.figure(figsize=(10,10))
dpi_val = 96
actual_pic_size = img_shape[0]/dpi_val
nplot = 11
for count in range(1,nplot):
    ax = fig.add_subplot(1,nplot,count)
    index = np.random.randint(0,len(data))
    plt.axis('off')
    ax.imshow(data[index])

plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig(filename, bbox_inches = 'tight',pad_inches = 0)
plt.show()