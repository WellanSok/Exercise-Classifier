import dash
from dash.dependencies import Output, Input
from dash import dcc, html
from datetime import datetime
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request
import time
import sys
import copy
import pickle
from features import extract_features # make sure features.py is in the same directory
from util import reorient, reset_vars
import labels
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.signal import butter, freqz, filtfilt, firwin, iirnotch, lfilter, find_peaks

with open('classifier.pickle', 'rb') as f:
    classifier = pickle.load(f)

class_names = labels.activity_labels

window_size = 100 # ~1 sec assuming 100 Hz sampling rate
step_size = 100  # no overlap
index = 0 # to keep track of how many samples we have buffered so far

def predict(window):
    """
    Given a window of accelerometer data, predict the activity label. 
    """
	
    # TODO: extract features over the window of data
    X = []
    feature_names, x = extract_features(window)
    X.append(x)
    X = np.asarray(X)
    # TODO: use classifier.predict(feature_vector) to predict the class label.
    # Make sure your feature vector is passed in the expected format
    prediction = classifier.predict(X)
    predictIndex = int(prediction[0])
    print(f"Predicted class: {class_names[predictIndex]}")
    # TODO: get the name of your predicted activity from 'class_names' using the returned label.
    # return the activity name.
    
    return class_names[predictIndex]

print("Loading data...")
sys.stdout.flush()
data_file = 'data/gibTestData.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))

def butterFilt(data):
    # Filter requirements.
    order = 3 
    fs = 100  # sample rate, Hz
    cutoff = 1.5  # desired cutoff frequency of the filter, Hz. MODIFY AS APPROPROATE
    # Create the filter.

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b,a,data)
    return filtered

#smooth accel_x,y,z
data[:,2] = butterFilt(data[:,2]) 
data[:,3] = butterFilt(data[:,3])
data[:,4] = butterFilt(data[:,4])
#smooth gyro_x,y,z
data[:,5] = butterFilt(data[:,5])
data[:,6] = butterFilt(data[:,6])
data[:,7] = butterFilt(data[:,7])

actArray = []
print("This code actually ran")
for i in range(0,len(data),window_size):
	subset = data[i:i+window_size]
	actArray.append(predict(np.asarray(subset)))
print(f"Seconds doing pushups: {actArray.count('pushup')} \n Seconds doing pullups: {actArray.count('pullup')} \n Seconds doing situps: {actArray.count('situp')} \n Seconds planking: {actArray.count('plank')}")
