import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
# network includes
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# path to the recording of the simulator
#recording_path = '/opt/my_training_data'
recording_path = '/home/workspace/CarND-Behavioral-Cloning-P3/my_training_data'

# image normalization
def per_image_standardization(x):
    import tensorflow as tf
    return tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)


# load csv
column_names = ['center', 'left', 'right',
                'steering', 'throttle', 'brake', 'speed']
lines = pd.read_csv(recording_path + '/driving_log.csv', names=column_names)

# shuffle
# this step must be done here as we want to select random elements for training
n_histogram_bins = 50
lines = lines.sample(frac=1).reset_index(drop=True)
histogram, bin_bounds = np.histogram(lines['steering'], bins=n_histogram_bins)
histogram.fill(0)

images = []
measurements = []

print("iterating through training data")
# iterate through training data
for idx, line in lines.iterrows():
    # load steering measurments
    steering_measurement = float(line[3])

    # steering correction for left and right camera images
    correction = 0.05

    # iterate through camera images (left, right, straight)
    for image_idx in range(3):
        source_path = line[image_idx]

        # no need to change path name as recording and training is done on the same machine
        filename = source_path.split('/')[-1]
        current_path = recording_path + '/IMG/' + filename

        # get steering measurement
        steering = steering_measurement

        # if left image, steer to the left (correction)
        if image_idx == 1:
            steering = steering_measurement + correction

        # if right image steer to the right (correction)
        if image_idx == 2:
            steering = steering_measurement - correction

        measurements.append(steering)

        # load the image
        image = cv2.imread(current_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

        # add additional flipped images
        flippedImage = np.fliplr(image)
        images.append(flippedImage)
        measurements.append(- steering)

X_train = np.array(images)
y_train = np.array(measurements)


input_shape= X_train.shape[1:]
nb_epoch = 20
batch_size = 10


#
# Model
#
model = Sequential()

# cropping
# keep only the road
model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=input_shape))

# normalize data and mean centering
# model.add(Lambda(lambda  x: x / 255.0 - 0.5))
model.add(Lambda(per_image_standardization))

# use model from nvidia
# https://devblogs.nvidia.com/deep-learning-self-driving-cars/
# Convolutional layer with 2x3 stride and 5x5 kernel
model.add(Conv2D(24, 5, 5, subsample=(2,2), border_mode='valid', activation="elu"))
model.add(Conv2D(36, 5, 5, subsample=(2,2), border_mode='valid', activation="elu"))
model.add(Conv2D(48, 5, 5, subsample=(2,2), border_mode='valid', activation="elu"))

# Convolutional layer without stride and 3x3 kernel
model.add(Conv2D(64, 3, 3, border_mode='valid', activation="elu"))
model.add(Conv2D(64, 3, 3, border_mode='valid', activation="elu"))

# Fully connected layer
model.add(Flatten())

model.add(Dense(100))
model.add(Dropout(0.2))

model.add(Dense(50))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Dense(1))

# mse = mean squared error
model.compile(loss='mse', optimizer='adam')

# model fit
history_object = model.fit(X_train, y_train, batch_size=batch_size, validation_split=0.2, shuffle=True, nb_epoch=nb_epoch, verbose=1)

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
        
print(model.summary())
# model save
model.save("model_with_dropout_3.h5")

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')
