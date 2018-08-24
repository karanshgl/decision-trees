import pandas as pd
import numpy as np
from tree import DecisionTree
from preprocess import save_random_set

save_random_set(1000, True)
save_random_set(1000, False)

train_set = pd.read_csv('./data/train.csv', delimiter = ',', header = None)
train_set = np.array(train_set)

train_labels = pd.read_csv('./data/train_labels.csv', delimiter = ',', header = None)
train_labels = np.array(train_labels)

test_set = pd.read_csv('./data/test.csv', delimiter = ',', header = None)
test_set = np.array(test_set)

test_labels = pd.read_csv('./data/test_labels.csv', delimiter = ',', header = None)
test_labels = np.array(test_labels)


tree = DecisionTree()
tree.fit(train_set, train_labels)
error = 0
for index in range(test_set.shape[0]):
	if tree.get_label(test_set[index]) == test_labels[index]:
		error += 1
print(error*1.0/test_set.shape[0])

# tree.save('./model/tree.pkl')