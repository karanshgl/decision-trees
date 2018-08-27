import pandas as pd
import numpy as np
from decision_tree import DecisionTree
from random_forest import RandomForest
from utils import add_noise
from preprocess import save_train_set, save_test_set

# save_train_set(1000, keep_validation = 0.5)
# save_test_set(1000)

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


# noise = [0.005, 0.01, 0.05, 0.1]
# for n in noise:
# 	tl = add_noise(train_labels, n)
# 	tree = DecisionTree()
# 	tree.fit(train_set, tl)
# 	print("Train: ", tree.accuracy(train_set, train_labels))
# 	print("Validation: ", tree.accuracy(validation_set, validation_labels))
# 	print("Test: ", tree.accuracy(test_set, test_labels))
# 	tree.save('./model/noise/tree_'+str(n)+'.pkl')
# tree.fit(train_set, train_labels)
# print("Before Pruning")
# print("Train: ", tree.accuracy(train_set, train_labels))
# print("Validation: ", tree.accuracy(validation_set, validation_labels))
# print("Test: ", tree.accuracy(test_set, test_labels))
# tree.save('./model/early_stopping/tree_ig.pkl')
# print(tree.height)
# print(tree.leaves)

# xy_nodes = []
# xy_acc = []

# for i in range(51):
# 	tree = DecisionTree(max_height = i)
# 	tree.fit(train_set, train_labels)
# 	xy_nodes.append((int(i),tree.leaves))
# 	xy_acc.append((int(i),tree.accuracy(test_set, test_labels)))
# 	print("{} Done {}".format(i, tree.leaves))

# xy_nodes = np.array(xy_nodes)
# xy_acc = np.array(xy_acc)
# np.savetxt("./experiments/graphs/height_vs_leaves.csv", xy_nodes, delimiter=",")
# np.savetxt("./experiments/graphs/height_vs_accu.csv", xy_acc, delimiter=",")

# tree.prune(validation_set, validation_labels)
# print("After Pruning")
# print("Train: ", tree.accuracy(train_set, train_labels))
# print("Validation: ", tree.accuracy(validation_set, validation_labels))
# print("Test: ", tree.accuracy(test_set, test_labels))

# error = 0
# for index in range(validation_set.shape[0]):
# 	if tree.predict(validation_set[index]) == validation_labels[index]:
# 		error += 1
# print("Val: ", error*1.0/validation_set.shape[0])
# error = 0
# for index in range(test_set.shape[0]):
# 	if tree.predict(test_set[index]) == test_labels[index]:
# 		error += 1

# print("Test: ", error*1.0/test_set.shape[0])
# print("Height is ", tree.height)
# print("Leaves is ", tree.leaves)

# tree.prune(validation_set, validation_labels)
# tree.save('./model/tree.pkl')
# error = 0
# for index in range(validation_set.shape[0]):
# 	if tree.predict(validation_set[index]) == validation_labels[index]:
# 		error += 1
# print("Val: ", error*1.0/validation_set.shape[0])
# error = 0
# for index in range(test_set.shape[0]):
# 	if tree.predict(test_set[index]) == test_labels[index]:
# 		error += 1

# print("Test: ", error*1.0/test_set.shape[0])
# print("Height is ", tree.height)
# print("Leaves is ", tree.leaves)


# forest = RandomForest(tree_count = 20, feature_bagging = 325)
# forest.fit(train_set, train_labels)
# forest.save('./model/forest.pkl')
# error = 0
# for index in range(test_set.shape[0]):
# 	if forest.predict(test_set[index]) == test_labels[index]:
# 		error += 1

# print(error*1.0/test_set.shape[0])
