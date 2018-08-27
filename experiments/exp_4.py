# Add Module Path
import sys
import pickle
sys.path.append('../')

with open('../bin/w2b.pkl', 'rb') as fp:
	w2b = pickle.load(fp)
with open('../bin/bins.pkl', 'rb') as fp:
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
acc_fun_val = tree.prune(test_set, test_labels)

sub_acc = [(int(i+1),val) for i,val in enumerate(acc_fun_val)]
sub_acc = np.array(sub_acc)
np.savetxt('./graphs/subtree_vs_test_acc.csv', sub_acc, delimiter=",")