# Add Module Path
import sys
import pickle
sys.path.append('../../')

with open('../../bin/w2b.pkl', 'rb') as fp:
	w2b = pickle.load(fp)
with open('../../bin/bins.pkl', 'rb') as fp:
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
import argparse

###################################################################################################
# Load the data
train_set = pd.read_csv('../../data/train.csv', delimiter = ',', header = None)
train_set = np.array(train_set)

train_labels = pd.read_csv('../../data/train_labels.csv', delimiter = ',', header = None)
train_labels = np.array(train_labels)

validation_set = pd.read_csv('../../data/validation.csv', delimiter = ',', header = None)
validation_set = np.array(validation_set)

validation_labels = pd.read_csv('../../data/validation_labels.csv', delimiter = ',', header = None)
validation_labels = np.array(validation_labels)

test_set = pd.read_csv('../../data/test.csv', delimiter = ',', header = None)
test_set = np.array(test_set)

test_labels = pd.read_csv('../../data/test_labels.csv', delimiter = ',', header = None)
test_labels = np.array(test_labels)

feature_count = train_set.shape[1]

###################################################################################################

###################################################################################################


xy_nodes = []
xy_acc = []
xy_train = []
for i in range(51):
	tree = DecisionTree(max_height = i)
	tree.fit(train_set, train_labels)
	xy_nodes.append((int(i),tree.leaves))
	xy_acc.append((int(i),tree.accuracy(test_set, test_labels)))
	xy_train.append((int(i), tree.accuracy(train_set, train_labels)))
	print("{} Done {}".format(i, tree.leaves))

xy_nodes = np.array(xy_nodes)
xy_acc = np.array(xy_acc)
xy_train = np.array(xy_train)
np.savetxt("./height_vs_leaves.csv", xy_nodes, delimiter=",")
np.savetxt("./height_test_acc.csv", xy_acc, delimiter=",")
np.savetxt("./height_train_acc.csv", xy_train, delimiter=",")