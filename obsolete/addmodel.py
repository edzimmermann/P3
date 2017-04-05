import os
import csv
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, ELU
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

import tensorflow as tf

## 1. Prepare data: read and clean-up paths, data etc.

## The simulator stores paths to images and angles in a CSV file
## Format is: center_image_path,left_image_path,right_image_path,center_angle,left_angle,right_angle
CENTER_CAMERA = 0
LEFT_CAMERA   = 1
RIGHT_CAMERA  = 2
CENTER_ANGLE  = 3
LEFT_ANGLE    = 4
RIGHT_ANGLE   = 5
MAX_CAMERAS   = 3 ## Center, Left and Right cameras
## NOTE: Angle in CSV is camera + numer_of_cameras (currently 3)


# Save filepaths to driving_log array 
## Clean up paths to use from the current directory

def fix_path (source_path, directory = "."):
    ## This strategy lets us also move the recorded directories without
    ## hurting anything
    if (source_path[1] != ':'):
       filename = os.path.split (source_path)[1]
    else:
       filename = source_path.split('\\')[-1] ## Since we collected on a Windows machine
    # fixed_path = os.path.join(directory, "IMG")
    # fixed_path = os.path.join(fixed_path, filename)
    fixed_path = directory + '/IMG/' + filename ## Read on Unix and MS
    # print ("PATH: ", fixed_path)
    return fixed_path


driving_log = []
def append_driving_log(directory, lines = driving_log):
    ## From Udacity vioeo
    ## Format is:   path_center, path_left, path_right, angle, angle, angle
    csv_filepath = os.path.join( directory, 'driving_log.csv')
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            line[0] = fix_path(line[0], directory) ## Center
            line[1] = fix_path(line[1], directory) ## Left
            line[2] = fix_path(line[2], directory) ## Right
            lines.append(line)
    return lines 

## Examples should get the paths as "IMG/..."
driving_log = append_driving_log('run1')
driving_log = append_driving_log('run2')
driving_log = append_driving_log('run3')

## Sanity check
## print("Line1: ", driving_log[0])

framerate=24 
frames = len(driving_log)
print("Driving log: ", frames, " images ", frames // framerate, " sec.")        

# Split into training and validation sets
training_set, validation_set = train_test_split(driving_log, test_size=0.2)


## Hint from class
def limit(X, y, beta=.08, alpha=.5):  
    # near 0 angles
    bad = [k for k,v in enumerate(y) if v >=-beta and v <= beta]
    # larger angles 
    good = list(set(range(0, len(y)))-set(bad))
    n = len(bad)
    new = good + [bad[i] for i in np.random.randint(0, n, int(n*alpha))]
    return X[new,], y[new]


## Default is Center camera
def myGenerator(samples, batch_size=32, camera = -1):
  
    ## Choice camera at random 1/3 of the time unless specified
    if (camera < 0):
        if (np.random.choice(3) == 0):
            camera = np.random.choice(MAX_CAMERAS)
        else: ## Rest of the time center camera
            camera = CENTER_CAMERA
    # assert (camera < MAX_CAMERAS) ## Don't want junk

    num_items = len(samples)
    print ("Loading {} images".format(num_items))
    while True:
        shuffle(samples)
        for offset in range(0, num_items, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_samples:
                ## USE Center Image and Angles
                image = line[camera] ## path to image
                angle = float(line[camera+MAX_CAMERAS])
                ## print ("Load: '", image, "' @ ", angle)
                img = mpimg.imread(image) ## read image
                ## Sometimes flip image (Udacity Data Augmentation suggestion)
                if ((camera == CENTER_CAMERA) and np.random.choice(2)) :
                   ## We only want to flip the center camera
                   images.append ( np.fliplr(img) )
                   angles.append ( -angle )
                else:
                   images.append(img)
                   angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            # X_train, y_train = limit (images, angles, 0.5) 
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
training_generator   = myGenerator(training_set)
validation_generator = myGenerator(validation_set)


## 2. Data Preprocessing functions

def resize_image(image):
    import tensorflow as tf
    return tf.image.resize_images(image, (40, 160))


##############################################################
## 3. Model (data preprocessing incorporated into model)
##############################################################

# Model (NVIDIA)

model = Sequential()

## Crop 70 pixels from the top of the image and 20 from the bottom
model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))

# Resize the data
model.add(Lambda(resize_image))
# Normalise the data
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# input_shape = (160, 320, 3)

# model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape))

# Conv layer 1
model.add(Convolution2D(24, 5, 5, subsample=(4, 4), border_mode="same"))
model.add(ELU())
# Conv layer 2
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
# Conv layer 3
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
# Conv Layer 4
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same"))
model.add(ELU())
# Conv layer 5
model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode="same"))
model.add(ELU())
##
model.add(Flatten())
## Connected Layer 1
model.add(Dense(100))
model.add(ELU())
## Connected Layer 2
model.add(Dense(50))
model.add(ELU())
##
model.add(Dropout(0.3))
##
## Connected layer 3
model.add(Dense(10))
model.add(ELU())
#
model.add(Dropout(.5))
## Final
model.add(Dense(1))

adam = Adam(lr=0.0001)

model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

print("Model summary:\n", model.summary())

##############################################
## 4. Train
##############################################
batch_size = 32
nb_epoch =  10 ## 20 

# Save model weights after each epoch
checkpointer = ModelCheckpoint(filepath="./tmp/v2-weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=False)

# Train model using generator
model.fit_generator(training_generator,
                    samples_per_epoch=len(training_set), 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_set), nb_epoch=nb_epoch,
                    callbacks=[checkpointer])

##############################################
## 5. Save 
##############################################

# https://github.com/udacity/sdc-issue-reports/issues/333
model_json = model.to_json()
with open("./model.json", "w") as outfile:
    outfile.write(model_json)
model.save_weights("./model.h5")
print ("Done")




def load_model():
    model = model_from_json(open('model.json').read())
    model.load_weights('./model.h5')
    model.compile(optimizer=rmsprop, loss='mse')
    return model


def save_model(model):    
    json_string = model.to_json()
    open('model.json', 'w').write(json_string)
    model.save_weights('./model.h5', overwrite=True)
