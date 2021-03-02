#MNIST CNN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from keras.models import Sequential
from keras.optimizers import SGD
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
import time

print(tf.__version__)
print(keras.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.compat.v1.Session(config=config) 
#Note: To run on gpu/cpu (uses gpu by default)
# with tf.device("gpu:0"):
#    print("tf.keras code in this scope will run on GPU")

# with tf.device("cpu:0"):
#    print("tf.keras code in this scope will run on CPU")

num_classes = 10
input_shape = (32,32,3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test,num_classes)
#Scale image from 0,1 range (Do I really have to do this manually?)
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

#x_train = tf.reshape(x_train, [32, 32, 3])
#x_test = tf.reshape(x_test, [32, 32, 3])

#Shape images to (28,28,1)
#x_train = np.expand_dims(x_train,-1)
#x_test = np.expand_dims(x_test, -1)

# Convert class vectors to binary class matrices
print("y_train samples: ", y_train.shape)
print("y_test samples: ", y_test.shape)
#y_train = keras.utils.to_categorical(y_train)
#y_test = keras.utils.to_categorical(y_test)

print("x_train shape: ", x_train.shape)
print("x_train[0] shape: ", x_train[0].shape)
print("x_train samples: ", x_train.shape[0])
print("x_test samples: ", x_test.shape[0])
print("y_train samples: ", y_train.shape)
print("y_test samples: ", y_test.shape)

# model = keras.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation="relu", input_shape=input_shape))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Flatten())
# model.add(layers.Dense(128,activation="relu"))
# model.add(layers.Dense(num_classes,activation="softmax"))
#opt = SGD(lr=0.01, momentum=0.9)

model = keras.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
# compile model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size = 64
epochs = 10

start = time.time()

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

end = time.time()
print(end - start)