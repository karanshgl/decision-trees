from decision_tree import DecisionTree
from collections import defaultdict
import numpy as np
import pickle

class RandomForest:
	"""
	Class that represents a decision forest

	Paramters
	tree_count: int: number of decision trees
	instance_bagging: int: number of instances to take at a time
	feature_bagging: int: number of features to take at a time
	max_height: max_height limit on every tree
	"""

	def __init__(self, tree_count = 10, instance_bagging = None, feature_bagging = None, max_height = None, ig_threshold = None):

		self.trees = [DecisionTree(max_height = max_height, ig_threshold = ig_threshold) for i in range(tree_count)]
		self.tree_features = []
		self.instance_bagging = instance_bagging
		self.feature_bagging = feature_bagging


	def save(self, file_name):
		"""
		Saves the self
		"""
		with open(file_name,'wb') as fp:
			pickle.dump(self, fp)

	def load(self, file_name):
		"""
		Loads the model to the self
		"""
		with open(file_name, 'rb') as fp:
			return pickle.load(fp)


	def fit(self, x_vals, y_vals):


		for tree in self.trees:

			# Get the indices of the instance matrixes
			rows, cols = np.arange(x_vals.shape[0]), np.arange(x_vals.shape[1]) 

			if self.instance_bagging:
				# Randomly Select Instances
				rows = np.random.choice(x_vals.shape[0], self.instance_bagging)
			if self.feature_bagging:
				# Randomly Select Features
				cols = np.random.choice(x_vals.shape[1], self.feature_bagging)

			# Save the indices of features used
			self.tree_features.append(cols)

			# Create the instances to feed to your decision tree
			x = x_vals[rows,:]
			x = x[:,cols]
			y = y_vals[rows]
			tree.fit(x,y)


	def fit_with_pruning(self, x_vals, y_vals, vx_vals, vy_vals):

		for tree in self.trees:

			# Get the indices of the instance matrixes
			rows, cols = np.arange(x_vals.shape[0]), np.arange(x_vals.shape[1]) 

			if self.instance_bagging:
				# Randomly Select Instances
				rows = np.random.choice(x_vals.shape[0], self.instance_bagging)
			if self.feature_bagging:
				# Randomly Select Features
				cols = np.random.choice(x_vals.shape[1], self.feature_bagging)

			# Save the indices of features used
			self.tree_features.append(cols)

			# Create the instances to feed to your decision tree
			x = x_vals[rows,:]
			x = x[:,cols]
			y = y_vals[rows]
			tree.fit(x,y)
			tree.prune(vx_vals[:,cols], vy_vals)


	def predict(self, X):

		label = 0

		for i,tree in enumerate(self.trees):
			label += tree.predict(X[self.tree_features[i]])

		return 1 if label>=0 else -1


	def accuracy(self, feature_matrix, labels):
		"""
		Returns the accuracy of the tree on a feature matrix
		"""

		accuracy = 0
		for i,instance in enumerate(feature_matrix):
			if self.predict(instance) == labels[i]:
				accuracy += 1

		return accuracy*1.0/labels.shape[0]

