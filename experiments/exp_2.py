# Add Module Path
import sys
import pickle
sys.path.append('../')

with open('../bin/w2b.pkl', 'rb') as fp:
	w2b = pickle.load(fp)
with open('../bin/bins.pkl', 'rb') as fp:
	bins = pickle.load(fp)

"""
---------------------------------------------------------------------------------------------------
Use the training set to learn a decision tree. Discuss the statistics of the learned tree, for
example, effect of early stopping on the number of terminal nodes, effect of early stopping on
the prediction accuracy on the training dataset and the left out test dataset, attributes that are
most frequently used as split functions in the internal nodes of the tree, etc.
----------------------------------------------------------------------------------------------------
"""

from decision_tree import DecisionTree
import numpy as np
import pandas as pd

###################################################################################################
# Load the data
train_set = pd.read_csv('../data/train.csv', delimiter = ',', header = None)
train_set = np.array(train_set)

train_labels = pd.read_csv('../data/train_labels.csv', delimiter = ',', header = None)
train_labels = np.array(train_labels)

validation_set = pd.read_csv('../data/validation.csv', delimiter = ',', header = None)
validation_set = np.array(validation_set)

validation_labels = pd.read_csv('../data/validation_labels.csv', delimiter = ',', header = None)
validation_labels = np.array(validation_labels)

test_set = pd.read_csv('../data/test.csv', delimiter = ',', header = None)
test_set = np.array(test_set)

test_labels = pd.read_csv('../data/test_labels.csv', delimiter = ',', header = None)
test_labels = np.array(test_labels)

feature_count = train_set.shape[1]

###################################################################################################

###################################################################################################

# Statistics of Learnt Tree
tree = DecisionTree().load('../model/tree.pkl')

# Without Early Stopping
print("-----------------------------")
print("Without Early Stopping")
print("-----------------------------")
print("Height: {} | Terminal Nodes: {}".format(tree.height,tree.leaves))
print("Train Accuracy: ", tree.accuracy(train_set, train_labels))
print("Validation Accuracy: ", tree.accuracy(validation_set, validation_labels))
print("Test Accuracy: ", tree.accuracy(test_set, test_labels))
print("Number of times an attribute is used as the splitting function:")
print("Feature (Polarity Bin): Frequency")
for feature in range(feature_count):
	print('%.2f'%bins[feature],': ', tree.attribute_frequency[feature])


print("-----------------------------")
print("-----------------------------")

# With Early Stopping
# Height = 40
tree = DecisionTree().load('../model/early_stopping/tree_height.pkl')
print("Early Stopping by Max Height = 40")
print("-----------------------------")
print("Height: {} | Terminal Nodes: {}".format(tree.height,tree.leaves))
print("Train Accuracy: ", tree.accuracy(train_set, train_labels))
print("Validation Accuracy: ", tree.accuracy(validation_set, validation_labels))
print("Test Accuracy: ", tree.accuracy(test_set, test_labels))
print("Number of times an attribute is used as the splitting function:")
print("Feature (Polarity Bin): Frequency")
for feature in range(feature_count):
	print('%.2f'%bins[feature],': ', tree.attribute_frequency[feature])

print("-----------------------------")
print("-----------------------------")
# IG Threshold = 1e-2
tree = DecisionTree().load('../model/early_stopping/tree_ig.pkl')
print("Early Stopping by Information Gain Threshold = 0.01")
print("-----------------------------")
print("Height: {} | Terminal Nodes: {}".format(tree.height,tree.leaves))
print("Train Accuracy: ", tree.accuracy(train_set, train_labels))
print("Validation Accuracy: ", tree.accuracy(validation_set, validation_labels))
print("Test Accuracy: ", tree.accuracy(test_set, test_labels))

print("Number of times an attribute is used as the splitting function:")
print("Feature (Polarity Bin): Frequency")
for feature in range(feature_count):
	print('%.2f'%bins[feature],': ', tree.attribute_frequency[feature])

print("-----------------------------")
print("-----------------------------")

