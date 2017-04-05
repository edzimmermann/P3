import os
import csv
#import cv2
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

nb_epoch = 3 ## 10 ## 20

## 1. Prepare data: read and clean-up paths, data etc.

## The simulator stores paths to images and angles in a CSV file
## Format is: center_image_path,left_image_path,right_image_path,steering_angle,throttle,brake,speed
CENTER_CAMERA = 0
LEFT_CAMERA   = 1
RIGHT_CAMERA  = 2
STEERING      = 3
THROTTLE      = 4
BRAKE         = 5
SPEED         = 6
MAX_CAMERAS   = 3 ## Center, Left and Right cameras

MODEL_NAME="model"

def json_file_name(model_name):
    return model_name + ".json"
def weights_file_name(model_name):
    return model_name + ".h5"

## Use Mean Squaed Error and Adam optimier
def load_model(model_name):
    json = json_file_name(model_name)
    try:
        if (os.path.isfile(json)):
            print ("Loading...")
            with open( json_file_name(model_name), "r") as infile:
                model = model_from_json(infile.read())
                model.load_weights(weights_file_name(model_name))
                model.compile(optimizer='adam', loss='mse', metrics=['accuracy']) ## rmsprop
                return model
    except:
        print("Load error encountered with '{}'".format(json))
    return None

def save_model(model_name):
    json_string = model.to_json()
    with open(json_file_name(model_name), "w") as outfile:
        outfile.write(json_string)
    model.save_weights(weights_file_name(model_name))
    print ("\nSaved ", model_name)


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

def is_noise(line):
    """
    Examine lines in the CSV that we don't want to use 
    """
    ## Check angle
    angle = float(line[STEERING])
#   Tried to remove zero angles but this led to zig-zag driving
#    if (angle < 0.01) and (angle > -0.01):
#        return True
    ## Check speed
    speed = float(line[SPEED])
    if (speed < 15):
        return True


def append_driving_log(directory, lines = driving_log):
    ## From Udacity vioeo
    ## Format is:   path_center, path_left, path_right, angle, angle, angle
    csv_filepath = os.path.join( directory, 'driving_log.csv')
    added = 0
    rejected = 0
    with open(csv_filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if (is_noise(line)):
              rejected += 1
              continue
            for i in range (0, MAX_CAMERAS):
                line[i] = fix_path(line[i], directory) ## 
            ## Smooth steering angle
            # angle = float(line[STEERING])
            ## Don't load where steering wheel is under 0.01 degrees
            added += 1
            lines.append(line)
    if (added > 0):
        print ("Added {} captures (x 3 cameras) from ".format(added), directory)
        print ("Rejected ", rejected)
    return lines 

## Examples should get the paths as "IMG/..."
# driving_log = append_driving_log('run1')
#driving_log = append_driving_log('run2')
#driving_log = append_driving_log('run3')
#driving_log = append_driving_log("run_rev") ## This run ran counter clockwize
driving_log = append_driving_log("extra")

## Sanity check
## print("Line1: ", driving_log[0])

framerate=60 # default, normally in film 24 or 25
frames = len(driving_log)
print("Total driving log: ", frames, " captures ", frames // framerate, " sec.")        

# Split into training and validation sets
training_set, validation_set = train_test_split(driving_log, test_size=0.2)


###################################################################
##
##  Generator for fit_generator
##
###################################################################
## Default is Center camera
DEFAULT_BATCH=64
bias = np.array([0.0, 0.25, -0.25]) ## CENTER LEFT RIGHT

def myGenerator(samples, training=True, batch_size=DEFAULT_BATCH, camera = -1):
  
    num_items = len(samples)
    print ("Loading {} captures".format(num_items))
    while True:
        shuffle(samples)
        for offset in range(0, num_items, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for line in batch_samples:
                 angle = float(line[STEERING])
                 for i in range (0, MAX_CAMERAS):
                    img = mpimg.imread(line[i]) ## read image
                    images.append(img)
                    a = angle + bias[i]
                    angles.append (a)
                    ## We only flip image when the angle is non-zero
                    if training : # (i == CENTER_CAMERA or (np.random.choice(2) and  (angle > 0.01 or angle < -0.01))):
                        images.append( np.fliplr(img) )
                        angles.append ( -a) 
                    
                    
#                ## USE Center Image and Angles
#                ## When no camera use 50% of time random choice
#                if (camera < 0 or camera >= MAX_CAMERAS):
#                    if (np.random.choice(2) == 0):
#                        camera = np.random.choice(MAX_CAMERAS)
#                    else :
#                        camera = CENTER_CAMERA
#                image = line[camera] ## path to image
#                angle = float(line[STEERING]) + bias[camera]
#                # print ("Load: '", image, "' @ ", angle)
#                img = mpimg.imread(image) ## read image
#                ## Sometimes flip image (Udacity Data Augmentation suggestion)
#                if ((camera == CENTER_CAMERA) and (angle > 0.03 or angle < -0.03)): # and  np.random.choice(2)) :
#                   ## We only want to flip the center camera
#                   images.append ( np.fliplr(img) )
#                   angles.append ( -angle )
#                # else:
#                images.append(img)
#                angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
training_generator   = myGenerator(training_set, True)
validation_generator = myGenerator(validation_set, False)


## 2. Data Preprocessing functions

##### Crop car hood and sky (looking at the images)
## NOTE: numbers chosen to also get a nice shape dimension
## input image is 160 x 320
CROP_TOP = 65
CROP_BOTTOM = 25
### Leaves 160 - 65 - 25 = 70 pixel 


##############################################################
## 3. Load or create model 
##############################################################


model = load_model(MODEL_NAME)
if (model == None):
# Model (NVIDIA)
    print ("Building network from scratch")
    model = Sequential()

    model.add(Cropping2D(cropping=((65,25), (0,0)), dim_ordering='tf', input_shape=(160,320,3)))
    ## 160 - 25 - 65 = 70 pixels 
    ## Output shape = (70, 320, 3)

    ## NVIDIA uses 66x200
    ## We have now 70x320

    # Normalise the data
    model.add(Lambda(lambda x: (x/255.0) - 0.5))

    ################ Model Design starts really here ###########
    ## https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    ## We use a modified network

    bm = "same" ## "valid" ## "same"
 
    # model.add(Convolution2D(3, 5, 5))
    # model.add(ELU())
    # Conv layer 1
    model.add(Convolution2D(24, 5, 5, subsample=(4, 4), border_mode=bm))
    model.add(ELU())
    # Conv layer 2
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode=bm))
    model.add(ELU())
    # Conv layer 3
    ## NVIDIA 
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode=bm))
    model.add(ELU())
    # Conv Layer 4
    ## NVIDIA 64@3x20
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode=bm))
    model.add(ELU())
    # Conv layer 5
    # model.add(Convolution2D(128, 3, 3, subsample=(1, 1), border_mode=bm))
    # model.add(ELU())
    ##
    model.add(Flatten())
    ## Connected Layer 1
    # model.add(Dense(1164)) 
    # model.add(ELU()) 
    ## Connected Layer 2
    model.add(Dense(100))
    model.add(ELU())
    ## Connected Layer 3
    model.add(Dense(50))
    model.add(ELU())
    ## Connected layer 4
    model.add(Dense(10))
    model.add(ELU())
    #
    model.add(Dropout(.5))
    ## Final
    model.add(Dense(1))

    adam = Adam(lr=0.0001)

    model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])

model.summary()

##############################################
## 4. Train using generator 
##############################################

# Train model using generator
steps = len(training_set)*4 // DEFAULT_BATCH
model.fit_generator(training_generator,
                    steps_per_epoch=steps, 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_set)*3, nb_epoch=nb_epoch,
                    callbacks=[])

##############################################
## 5. Save model
##############################################
save_model(MODEL_NAME)
print ("Done")
