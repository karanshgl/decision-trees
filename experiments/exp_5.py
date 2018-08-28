# Add Module Path
import sys
import pickle
sys.path.append('../')

with open('./bin/w2b.pkl', 'rb') as fp:
	w2b = pickle.load(fp)
with open('./bin/bins.pkl', 'rb') as fp:
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
from random_forest import RandomForest
import numpy as np
import pandas as pd

###################################################################################################
# Load the data
train_set = pd.read_csv('./data/train.csv', delimiter = ',', header = None)
train_set = np.array(train_set)

train_labels = pd.read_csv('./data/train_labels.csv', delimiter = ',', header = None)
train_labels = np.array(train_labels)

validation_set = pd.read_csv('./data/validation.csv', delimiter = ',', header = None)
validation_set = np.array(validation_set)

validation_labels = pd.read_csv('./data/validation_labels.csv', delimiter = ',', header = None)
validation_labels = np.array(validation_labels)

test_set = pd.read_csv('./data/test.csv', delimiter = ',', header = None)
test_set = np.array(test_set)

test_labels = pd.read_csv('./data/test_labels.csv', delimiter = ',', header = None)
test_labels = np.array(test_labels)

feature_count = train_set.shape[1]

###################################################################################################

###################################################################################################

test_acc = []
train_acc = []
trees = [0, 20, 40, 60, 80, 100]
def run():

	for i in trees:
		forest = RandomForest(tree_count = i+1, feature_bagging = 350)
		forest.fit(train_set, train_labels)
		a = forest.accuracy(test_set, test_labels)
		b = forest.accuracy(train_set, train_labels)
		test_acc.append((i+1,a))
		train_acc.append((i+1,b))
		print("Tree Count: {} | Train Accuracy: {} | Test Accuracy: {}".format(i+1,b,a))

# test_acc = np.array(test_acc)
# np.savetxt('./graphs/forest_vs_test_acc_1.csv', test_acc, delimiter = ",")
# train_acc = np.array(train_acc)
# np.savetxt('./graphs/forest_vs_train_acc_1.csv', train_acc, delimiter = ",")