import argparse
import base64
from datetime import datetime
import os
import shutil
import math
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model

import json

import time

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array



from keras.utils import plot_model
#import pydot_ng as pydot
import pydot

pydot.find_graphviz = lambda: True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Visualizhation')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model json file. h5 weights should be on the same path.'
    )
    args = parser.parse_args()


    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)


    plot_model(model, to_file='model.png')
