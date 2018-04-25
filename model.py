import numpy as np
import csv
import cv2
import tensorflow as tf

import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D




def generator(img_dir, angles, batch_size=16):
    num_samples = len(samples)
    correction = [0.0, 0.2, -0.2]
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    angle = float(batch_sample[3]) + correction[i]
                    images.append(image)
                    angles.append(angle)

            augmented_images, augmented_angles = [], []
            for image,angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield shuffle(X_train, y_train)



def loadfile(csv_dir, img_dir):
 
    with open(csv_dir) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images_dir = []
    angles = []
    correction = [0.0, 0.2, -0.2]
    for line in lines[1:]:
        for i in range(3):
            name = img_dir + line[i].split('/')[-1]
            angle = float(line[3]) + correction[i]
            images_dir.append(name)
            angles.append(angles)
    return images_dir, angles


img_dir, angles = loadfile('./data/data1/driving_log.csv', './data/data1/IMG/')
img_dir1, angles1 = loadfile('./data/data2/driving_log.csv', './data/data2/IMG/')
img_dir2, angles2 = loadfile('./data/data3/driving_log.csv', './data/data3/IMG/')
img_dir.extend(img_dir1)
img_dir.extend(img_dir2)
angles.extend(angles1)
angles.extend(angles2)


train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model = Sequential()
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(36,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(48,(5,5), strides=(2,2), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)

model.save('model.h5')
