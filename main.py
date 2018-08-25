import pandas as pd
import numpy as np
from decision_tree import DecisionTree
from random_forest import RandomForest
from preprocess import save_train_set, save_test_set

save_train_set(2000)
save_test_set(2000)

train_set = pd.read_csv('./data/train.csv', delimiter = ',', header = None)
train_set = np.array(train_set)

train_labels = pd.read_csv('./data/train_labels.csv', delimiter = ',', header = None)
train_labels = np.array(train_labels)

validation_set = pd.read_csv('./data/validation.csv', delimiter = ',', header = None)
validation_set = np.array(train_set)

validation_labels = pd.read_csv('./data/validation_labels.csv', delimiter = ',', header = None)
validation_labels = np.array(train_labels)

test_set = pd.read_csv('./data/test.csv', delimiter = ',', header = None)
test_set = np.array(test_set)

test_labels = pd.read_csv('./data/test_labels.csv', delimiter = ',', header = None)
test_labels = np.array(test_labels)


# tree = DecisionTree(max_height = 30)
# tree.fit(train_set, train_labels)
# error = 0
# for index in range(validation_set.shape[0]):
# 	if tree.get_label(validation_set[index]) == validation_labels[index]:
# 		error += 1
# print("Val: ", error*1.0/validation_set.shape[0])
# error = 0
# for index in range(test_set.shape[0]):
# 	if tree.get_label(test_set[index]) == test_labels[index]:
# 		error += 1

# print("Test: ", error*1.0/test_set.shape[0])
# print("Height is ", tree.height)
# print("Leaves is ", tree.leaves)

# tree.save('./model/tree.pkl')

forest = RandomForest(tree_count = 50, feature_bagging = 325, max_height = 50)
forest.fit(train_set, train_labels)
error = 0
for index in range(test_set.shape[0]):
	if forest.predict(test_set[index]) == test_labels[index]:
		error += 1

print(error*1.0/test_set.shape[0])
