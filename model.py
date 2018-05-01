import numpy as np
import csv
import cv2
import tensorflow as tf

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential,load_model
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout, ELU

# Function for crop and resize the image to 66x200 
def img_resize(img):
    new_img = img[50:140,:,:]
    new_img = cv2.resize(new_img, (200, 66), interpolation=cv2.INTER_AREA)
    return new_img


# Data generator
def generator(samples, batch_size=16):
    num_samples = len(samples)
    # Angle correction: +0.275 for left, -0.275 for right
    correction = [0.0, 0.275, -0.275]
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    # path for the image
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    # crop and resize the image
                    new_image = img_resize(image)
                    # angle correction
                    angle = float(batch_sample[3]) + correction[i]
                    # add image and angle to the lists
                    images.append(new_image)
                    angles.append(angle)

            # augment data by flipping images
            augmented_images, augmented_angles = [], []
            for image,angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)


# Read csv file
lines = []
csv_dir = './data/driving_log.csv'
with open(csv_dir) as csvfile:
    reader = csv.reader(csvfile)
    for line in list(reader)[1:]:
        lines.append(line)

# Split data to training samples and validation samples
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=12)
validation_generator = generator(validation_samples, batch_size=12)


# Create new Nvidia model
model = Sequential()
# Image normalization
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(66, 200, 3)))

# Convolution layers
model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))

# Flatten layer
model.add(Flatten())

# Fully connected layeers
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(1))


# Constant for determine whether train an old model
use_loaded = True

# Training the model
if use_loaded:
    # Load previous model
    loaded_model = load_model('model_pre.h5')
   # Old model
    history_object = loaded_model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose=1)
    loaded_model.save('model.h5')
else:
    # New model
    model.compile(optimizer='adam', loss='mse')
    history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose=1)
    model.save('model.h5')


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


