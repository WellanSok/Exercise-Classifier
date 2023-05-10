# -*- coding: utf-8 -*-
"""
This file is used for extracting features over windows of tri-axial accelerometer 
data. We recommend using helper functions like _compute_mean_features(window) to 
extract individual features.

As a side note, the underscore at the beginning of a function is a Python 
convention indicating that the function has private access (although in reality 
it is still publicly accessible).

"""

import numpy as np
from scipy import signal 
from scipy.signal import find_peaks

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)

# TODO: define functions to compute more features

def compute_std(window):
    """
    Computes the std over given window i used the comp mean function as template
    """
    std_w = np.std(window)
    return std_w

def rms(window):
    rms_w = np.sqrt(np.mean(np.square(window)))
    return rms_w

def domFreq(window):
    magnitude = np.linalg.norm(window, axis=1)
    # Compute the DFT of the magnitude signal
    dft = np.fft.rfft(magnitude).astype(float)
    # Compute the power spectrum of the DFT
    power_spectrum = np.square(np.abs(dft))

    # Find the dominant frequency of the signal
    dominant_freq_bin = np.argmax(power_spectrum)
    dominant_freq = np.fft.rfftfreq(len(magnitude))[dominant_freq_bin]

    return dominant_freq

# returns length 4 array of x y z and magnitude entropy 
def entropy(window):
    entropy = np.zeros(4)
        
    # Compute the histogram of the axis
    hist, bin_edges = np.histogram(window, bins=10, density=True)
        
    # Compute the entropy of the histogram
    prob = hist[hist != 0]
    entropy = -np.sum(prob * np.log2(prob))
    
    return entropy

def entropyMagnitude(window):
    magnitude = np.linalg.norm(window, axis=1)
    hist, bin_edges = np.histogram(magnitude, bins=10, density=True)
    
    # Compute the entropy of the magnitude histogram
    prob = hist[hist != 0]
    entropy = -np.sum(prob * np.log2(prob))

    return entropy

def orientation(window):
    # Assume the data is in the format [gyro_x, gyro_y, gyro_z]
    dt = 0.01  # sample interval (in seconds)
    roll = np.arctan2(window[:,1], window[:,2])
    pitch = np.arctan2(-window[:,0], np.sqrt(window[:,1]**2 + window[:,2]**2))
    yaw = np.cumsum(window[:,2] * dt)  # integrate the z-axis angular velocity to get yaw
    return np.column_stack((roll, pitch, yaw))

def jerk(window):
    dt = 0.01  # sample interval (in seconds)
    dacc = np.diff(window, axis=0) / dt
    jerk = np.sqrt(np.sum(dacc**2, axis=1))
    return jerk

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.
    
    """

    """
    Statistical
    These include the mean, variance and the rate of zero- or mean-crossings. The
    minimum and maximum may be useful, as might the median
    
    FFT features
    use rfft() to get Discrete Fourier Transform
    
    Entropy
    Integrating acceleration
    
    Peak Features:
    Sometimes the count or location of peaks or troughs in the accelerometer signal can be
    an indicator of the type of activity being performed. This is basically what you did in
    assignment A1 to detect steps. Use the peak count over each window as a feature. Or
    try something like the average duration between peaks in a window.
    """

    
    x = []
    feature_names = []
    win = np.array(window)

    x.append(_compute_mean_features(win[:,0]))
    feature_names.append("x_mean")

    x.append(_compute_mean_features(win[:,1]))
    feature_names.append("y_mean")

    x.append(_compute_mean_features(win[:,2]))
    feature_names.append("z_mean")

    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    x.append(compute_std(win[:,0]))
    feature_names.append("std_x")

    x.append(compute_std(win[:,1]))
    feature_names.append("std_y")

    x.append(compute_std(win[:,2]))
    feature_names.append("std_z")

    x.append(rms(win[:,0]))
    feature_names.append("rms_x")

    x.append(rms(win[:,1]))
    feature_names.append("rms_y")

    x.append(rms(win[:,2]))
    feature_names.append("rms_z")

    x.append(domFreq(win))
    feature_names.append("dominant freq")

    x.append(entropy(win[:,0]))
    feature_names.append("entropy_x")

    x.append(entropy(win[:,1]))
    feature_names.append("entropy_y")

    x.append(entropy(win[:,2]))
    feature_names.append("entropy_z")

    x.append(entropyMagnitude(win))
    feature_names.append("entropy_mag")

    feature_vector = list(x)
    return feature_names, feature_vector