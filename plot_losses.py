import numpy as np 
import matplotlib.pyplot as plt

x = np.genfromtxt("mini_64x64_losses.csv", delimiter=",")

length = x[-1,0]
values = list((range(len(x[:,2]))))
values = np.array(values)
NewRange = length
OldMin = values[0]
OldRange = values[-1] - values[0]
values = (((values - OldMin) * NewRange) / OldRange)
numVals = len(values)
print(length)
print(values)
plt.plot(x[:,2])
plt.plot(x[:,3])
plt.xticks([0,numVals/4,numVals/2,3*numVals/4,numVals, numVals],[1,2,3,4,5])
plt.show()
