import numpy as np
from utils import entropy, best_split
import pickle

class Node:
	"""
	Class that represents a node in a decision tree

	Parameters:
	feature_index: the index of the feature (word/bin) the node takes
	left: The left child of the tree
	right: The right child of the tree

	"""
	def __init__(self, fi = None, left_side = None, right_side = None):
		self.feature_index = fi
		self.left = left_side
		self.right = right_side
		self.split_val = None

		# Initially there is no label 
		self.label = None

	def put_label(self, label):
		"""
		Positive or Negative

		"""
		self.label = label


class DecisionTree:
	"""
	Class that represents a decision tree

	"""

	def __init__(self):

		# Initially the root would be none
		self.root = None

	def save(self, file_name):
		"""
		Saves the self
		"""
		with open(file_name,'wb') as fp:
			pickle.dump(self, fp)


	def fit(self, x_vals, y_vals):
		"""
		Fits the instances x to their values y.

		Parameters:
		x_vals: instances, each instance value is a dictionary 
		with attribute name as the key and its frequency as value
		y_vals: the label of the instance
		vocab: a list of all attributes
		"""
		self.root = self._build_tree(x_vals, y_vals)


	def _build_tree(self, x_vals, y_vals):
		"""
		Returns the root of a decision tree

		Parameters:
		x_vals: instances, each instance value is a dictionary 
		with attribute name as the key and its frequency as value
		y_vals: the label of the instance
		"""

		root = Node()

		instances, features = x_vals.shape


		# Base Cases
		if y_vals.mean() == 1:
			root.put_label(1)
			return root
		elif y_vals.mean() == -1:
			root.put_label(-1)
			return root
		elif x_vals.shape[1] == 0:
			root.put_label((1 if y_vals.mean()>=0 else -1))
			return root

		# Spliting
		root_feature = None	# The best feature
		max_info_gain = 0	# Max value of information gain

		positive_y = np.where(y_vals == 1, 1, 0).sum()
		negative_y = y_vals.shape[0] - positive_y
		entropy_y = entropy(1.0*positive_y/y_vals.shape[0], 1.0*negative_y/y_vals.shape[0])
		x_left, y_left = [],[]
		x_right, y_right = [], []
		split_val = 0


		for feature in range(features):
			# For all features calculate information gain
	
			# Left side contains the indices of rows which contain the feature
			left_side  = np.argwhere(x_vals[:,feature] > 0)
			# Right side contains the indices of rows which doesn't contain the feature
			right_side = np.argwhere(x_vals[:, feature] == 0)
			# If splits cannot happen
			if left_side.shape[0] == 0 or right_side.shape[0] == 0: continue

			left_instances, left_labels   = x_vals[left_side[:,0]], y_vals[left_side[:,0]]
			right_instances, right_labels = x_vals[right_side[:,0]], y_vals[right_side[:,0]]

			# Left Side
			total_left = left_labels.shape[0]
			positive_left = np.where(left_labels == 1, 1, 0).sum()
			negative_left = total_left - positive_left

			entropy_left = entropy(1.0*positive_left/total_left, 1.0*negative_left/total_left)

			# Right Side
			total_right = right_labels.shape[0]
			positive_right = np.where(right_labels == 1, 1, 0).sum()
			negative_right = total_right - positive_right

			entropy_right = entropy(1.0*positive_right/total_right, 1.0*negative_right/total_right)

			# Information Gain
			total = total_left + total_right
			IG = entropy_y - ((1.0*total_left/total)*entropy_left + (1.0*total_right/total)*entropy_right)

			if(IG > max_info_gain):
				root_feature = feature
				max_info_gain = IG
				x_left, y_left = left_instances, left_labels
				x_right, y_right =  right_instances, right_labels


		if root_feature == None:
			# No feature is giving good information gain
			root.put_label((1 if y_vals.mean()>=0 else -1))
			return root


		root.feature_index = root_feature
		root.split_val = split_val
		# print(x_vals.shape[1])
		x_left = np.delete(x_left, root_feature, axis = 1)
		root.left = self._build_tree(x_left, y_left)
		x_right = np.delete(x_right, root_feature, axis = 1)
		root.right = self._build_tree(x_right, y_right)
		return root

	def _find_label(self, X, root):
		"""
		Tree traversal to get the label
		"""

		if root.label: return root.label

		feature = root.feature_index
		split_val = root.split_val
		if X[feature]:
			# It has the feature
			X = np.delete(X, feature)
			return self._find_label(X, root.left)
		else:
			# It doesn't have the feature
			X = np.delete(X, feature)
			return self._find_label(X, root.right)


	def get_label(self, X):
		"""
		Returns the label for the instance X
		"""

		return self._find_label(X, self.root)


	