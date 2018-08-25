from decision_tree import DecisionTree
import numpy as np

class RandomForest:
	"""
	Class that represents a decision forest

	Paramters
	tree_count: int: number of decision trees
	instance_bagging: int: number of instances to take at a time
	feature_bagging: int: number of features to take at a time
	max_height: max_height limit on every tree
	"""

	def __init__(self, tree_count = 10, instance_bagging = None, feature_bagging = None, max_height = None):

		self.trees = [DecisionTree(max_height = max_height) for i in range(tree_count)]
		self.tree_features = []
		self.instance_bagging = instance_bagging
		self.feature_bagging = feature_bagging

	def fit(self, x_vals, y_vals):

		for tree in self.trees:

			rows, cols = np.arange(x_vals.shape[0]), np.arange(x_vals.shape[1]) 

			if self.instance_bagging:
				rows = np.random.choice(x_vals.shape[0], self.instance_bagging)
			if self.feature_bagging:
				cols = np.random.choice(x_vals.shape[1], self.feature_bagging)

			self.tree_features.append(cols)
			x = x_vals[rows,:]
			x = x[:,cols]
			y = y_vals[rows]
			tree.fit(x,y)

	def predict(self, X):

		label = 0

		for i,tree in enumerate(self.trees):
			label += tree.get_label(X[self.tree_features[i]])

		return 1 if label>=0 else -1
