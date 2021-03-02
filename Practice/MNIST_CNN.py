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
input_shape = (28,28,1)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#Scale image from 0,1 range (Do I really have to do this manually?)
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

#Shape images to (28,28,1)
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape: ", x_train.shape)
print("x_train samples: ", x_train.shape[0])
print("x_test samples: ", x_test.shape[0])

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential()
#model.add(keras.Input(shape=input_shape))
model.add(layers.Conv2D(32,(3,3),activation="relu", input_shape=input_shape))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(100,activation="relu"))
model.add(layers.Dense(num_classes,activation="softmax"))
#opt = SGD(lr=0.01, momentum=0.9)


model.summary()

batch_size = 128
epochs = 3

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])