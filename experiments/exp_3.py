# Add Module Path
import sys
import pickle
# sys.path.append('../')

with open('./bin/w2b.pkl', 'rb') as fp:
	w2b = pickle.load(fp)
with open('./bin/bins.pkl', 'rb') as fp:
	bins = pickle.load(fp)

"""
Let us add some noise to the dataset and observe its effect on the decision tree. Add 0.5, 1, 5
and 10% noise to the dataset. You can add this noise by randomly switching the label of a
proportion of data points. Construct the decision tree for each noise setting and quantify the
complexity of the learned decision tree using the number of nodes in the tree. Document and
discuss your observations about the quality of the model learned under these noise conditions.
"""

from decision_tree import DecisionTree
import numpy as np
import pandas as pd
from utils import add_noise

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

noise_list = [0.005, 0.01, 0.05, 0.1]
nodes = []
path = './model/noise/tree_{}.pkl'
def run(retrain = True):

	for noise in noise_list:
		print("-----------------------------")
		print("Noise: {}".format(noise))

		tree = DecisionTree()
		if retrain:
			tl = add_noise(train_labels, noise)
			tree.fit(train_set, tl)
		else:
			tree = DecisionTree().load(path.format(noise))
		print("Height: {} | Terminal Nodes: {}".format(tree.height,tree.leaves))
		nodes.append((noise,tree.leaves))
		print("Train Accuracy: ", tree.accuracy(train_set, train_labels))
		print("Validation Accuracy: ", tree.accuracy(validation_set, validation_labels))
		test = tree.accuracy(test_set, test_labels)
		print("Test Accuracy: ", test)
		# acc.append((noise,test))
		print("-----------------------------")

# arr = np.array(nodes)
# np.savetxt("./graphs/noise_nodes.csv", arr, delimiter=",")



