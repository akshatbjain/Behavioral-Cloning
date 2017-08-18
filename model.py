import csv
import cv2
import numpy as np

import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

correction = 0.229

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = '../New-Data/IMG/' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    flip_image = cv2.flip(image, 1)
                    images.extend([image.transpose((2, 0, 1)), flip_image.transpose((2, 0, 1))])
                    #images.append(image.transpose((2, 0, 1)))

                    if (i == 0):
                        angle = float(batch_sample[3])
                        flip_angle = float(batch_sample[3])*-1.0
                    elif (i == 1):
                        angle = float(batch_sample[3]) + correction
                        flip_angle = float(batch_sample[3])*-1.0 - correction
                    elif (i == 2):
                        angle = float(batch_sample[3]) - correction
                        flip_angle = float(batch_sample[3])*-1.0 + correction
                    angles.extend([angle, flip_angle])
                    #angles.append(angle)

                #center_image = cv2.imread(batch_sample[0]).transpose((2, 0, 1))
                #left_image = cv2.imread(batch_sample[1]).transpose((2, 0, 1))
                #right_image = cv2.imread(batch_sample[2]).transpose((2, 0, 1))
                #images.append(center_image)
                #images.extend([center_image, left_image, right_image])

                #center_angle = float(batch_sample[3])
                #correction = 0.2
                #left_angle = center_angle + correction
                #right_angle = center_angle - correction
                #angles.append(center_angle)
                #angles.extend([center_angle, left_angle, right_angle])

                #augmented_images.append(cv2.flip(image, 1))
                #augmented_measurements.append(measurement*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = []
with open('../New-Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for sample in reader:
        samples.append(sample)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(3, 160, 320), output_shape=(3, 160, 320))) # x/127.5 - 1.
model.add(Cropping2D(cropping=((60, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(1164))
#model.add(Dropout(0.5))
model.add(Dense(100))
#model.add(Dropout(0.5))
model.add(Dense(50))
#model.add(Dropout(0.5))
model.add(Dense(10))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.summary()
model.compile(loss='mse', optimizer='adam')

continue_training = True
phase = 0
epochs = 5
while(continue_training):
    phase += 1

    history_object = model.fit_generator(train_generator, samples_per_epoch =
        len(train_samples)*6, validation_data =
        validation_generator,
        nb_val_samples = len(validation_samples)*6,
        nb_epoch=epochs, verbose=1)

    model.save('model_' + str(phase) + '.h5')

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

    print('Continue Training? [True/False]: ')
    user_input = input()

    if(user_input == 'False' or user_input == 'false' or user_input == 'f' or user_input == 'F'):
        continue_training = False

    if (continue_training):
        print('Number of epochs? [Enter a integer value]: ')
        epochs = int(input())
