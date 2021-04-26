import numpy as np
from matplotlib import pyplot as plt
from numpy import load


data = load('impressionism_128x128_augmented.npy')
fig = plt.figure(figsize=(30,10))
nplot = 8
for count in range(1,nplot):
    ax = fig.add_subplot(1,nplot,count)
    index = np.random.randint(0,len(data))
    ax.imshow(data[index])
plt.show()