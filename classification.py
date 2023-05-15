# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
import sklearn
import sklearn.metrics
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from features import extract_features
from util import slidingWindow, reorient, reset_vars
from sklearn.tree import DecisionTreeClassifier
import pickle

import labels


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'data/all_labeled_data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------



# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 150
step_size = 150

# sampling rate should be about 100 Hz (sensor logger app); you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,1] - data[0,1])
sampling_rate = n_samples / time_elapsed_seconds

print("Sampling Rate: " + str(sampling_rate))

# TODO: list the class labels that you collected data for in the order of label_index (defined in labels.py)
class_names = labels.activity_labels

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []
feature_names = []
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:]
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation
cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
accuracy_list = []
precision_list = []
recall_list = []
for train_index, test_index in cv.split(X):
    #do the stuff this loop iterates over the folds
    betterTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    betterTree.fit(X[train_index], Y[train_index])
    y_test = Y[test_index]
    y_train = betterTree.predict(X[test_index])
    print(sklearn.metrics.confusion_matrix(y_test, y_train))
    precision_list.append(sklearn.metrics.precision_score(y_test,y_train,average=None,zero_division=1))
    accuracy_list.append(sklearn.metrics.accuracy_score(y_test,y_train))
    recall_list.append(sklearn.metrics.recall_score(y_test,y_train, average=None,zero_division=1))

# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds

# The following code handles strange behavior regarding zero divisions in score calculations
i = 0
for element,element2 in zip(precision_list,recall_list):
    if len(element)<4:
        precision_list[i] = np.append(precision_list[i],1)
    if len(element2)<4:
        recall_list[i] = np.append(recall_list[i],1)
    i+=1

print(f"Accuracy:{np.average(accuracy_list)}")
print(f"Precision:{np.average(precision_list)}")
print(f"Recall:{np.average(recall_list)}")

# TODO: train the decision tree classifier on entire dataset
superTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
superTree.fit(X, Y)

# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(superTree, out_file='tree.dot', feature_names = feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
print("saving classifier model...")
with open('classifier.pickle', 'wb') as f:
     pickle.dump(superTree, f)