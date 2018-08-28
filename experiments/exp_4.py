# Add Module Path
import sys
import pickle
# sys.path.append('../')

with open('./bin/w2b.pkl', 'rb') as fp:
	w2b = pickle.load(fp)
with open('./bin/bins.pkl', 'rb') as fp:
	bins = pickle.load(fp)

"""
Produce a pruned tree corresponding to the optimal tree size by computing the prediction
accuracy on the test set. Use a post-pruning strategy to prune the tree that has been learned
without applying any early stopping criteria. Document the change in the prediction accuracy
as a function of pruning (reduction in the number of nodes or rules/clauses obtained from the
tree).
"""

from decision_tree import DecisionTree
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

def run(retrain = True):
	# Statistics of Learnt Tree
	tree = DecisionTree()
	if retrain == False:
		tree = DecisionTree().load('./model/tree.pkl')
	else:
		tree.fit(train_set, train_labels)
		
	print("-----------------------------")
	print("Post-Pruning by Node Removal")
	print("-----------------------------")
	acc_fun_val, nodes_removed, test_acc = tree.prune_val_test(validation_set, validation_labels, test_set, test_labels)

	# sub_acc = [(nodes_removed[i],val) for i,val in enumerate(acc_fun_val)]
	# sub_acc_t = [(nodes_removed[i],val) for i,val in enumerate(test_acc)]
	for i,val in enumerate(acc_fun_val):
		print("Number of nodes removed: {} | Validation Accuracy: {} | Test Accuracy: {}".format(nodes_removed[i],val, test_acc[i]))

	print("Training Set Accuracy: ",tree.accuracy(train_set, train_labels))
		
	print("-----------------------------")
	# sub_acc = np.array(sub_acc)
	# sub_acc_t = np.array(sub_acc_t)
	# np.savetxt('./graphs/nodes_removed_vs_test_acc_and_val_1.csv', sub_acc, delimiter=",")
	# np.savetxt('./graphs/nodes_removed_vs_test_acc_and_val_3.csv', sub_acc_t, delimiter=",")
	# tree.save('../model/pruning/tree_pruned.pkl')