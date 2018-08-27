import numpy as np
from utils import entropy
import pickle
from collections import deque, defaultdict

class Node:
	"""
	Class that represents a node in a decision tree

	Parameters:
	feature_index: the index of the feature (word/bin) the node takes
	left: The left child of the tree
	right: The right child of the tree

	"""
	def __init__(self, fi = None, left_side = None, right_side = None, parent = None):
		self.feature_index = fi
		self.left = left_side
		self.right = right_side
		self.parent = parent

		# Children Count
		self.positive_instances = 0
		self.negative_instances = 0
		self.children = 0

		# Initially there is no label 
		self.label = None

	def put_label(self, label):
		"""
		Positive or Negative

		"""
		self.label = label

	def update_children_count(self, positive, negative):
		self.positive_instances = positive
		self.negative_instances = negative
		self.children = positive + negative 


class DecisionTree:
	"""
	Class that represents a decision tree

	Paramters
	max_height: Limit on the height of the tree
	ig_threshold: Minimum increase in information gain required for splt

	"""

	def __init__(self, max_height = None, ig_threshold = None):

		# Initially the root would be none
		self.root = None
		self.max_height = max_height
		self.ig_threshold = ig_threshold
		self.leaves = 0
		self.height = 0

		# Freqeuncy of the attributes used
		self.attribute_frequency = defaultdict(int)

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
		"""
		Fits the instances x to their values y.

		Parameters:
		x_vals: instances, each instance value is a dictionary 
		with attribute name as the key and its frequency as value
		y_vals: the label of the instance
		vocab: a list of all attributes
		"""
		self.root = self._build_tree(x_vals, y_vals)
		self.height = self._height(self.root)
		self.leaves = self._leaves(self.root)


	def _build_tree(self, x_vals, y_vals, cur_height = 0):
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
		elif (self.max_height != None and cur_height >= self.max_height):
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

			if(IG > (max_info_gain + (self.ig_threshold if self.ig_threshold else 0))):
				root_feature = feature
				max_info_gain = IG
				x_left, y_left = left_instances, left_labels
				x_right, y_right =  right_instances, right_labels

		if root_feature == None:
			# No feature is giving good information gain
			root.put_label((1 if y_vals.mean()>=0 else -1))
			return root

		root.feature_index = root_feature
		self.attribute_frequency[root_feature] += 1

		# Build left and right subtrees
		# x_left = np.delete(x_left, root_feature, axis = 1)
		root.left = self._build_tree(x_left, y_left, cur_height+1)
		# x_right = np.delete(x_right, root_feature, axis = 1)
		root.right = self._build_tree(x_right, y_right, cur_height+1)
		# Add Parent
		root.left.parent = root
		root.right.parent = root

		# Add Children Count
		root.update_children_count(positive_y, negative_y)
		return root

	def _find_label(self, X, root):
		"""
		Tree traversal to get the label
		"""
		
		if root.label: return root.label

		feature = root.feature_index
		if X[feature]:
			# It has the feature
			# X = np.delete(X, feature)
			return self._find_label(X, root.left)
		else:
			# It doesn't have the feature
			# X = np.delete(X, feature)
			return self._find_label(X, root.right)


	def predict(self, X):
		"""
		Returns the label for the instance X
		"""

		return self._find_label(X, self.root)


	def _height(self, root):
		"""
		Returns the height of the tree
		"""
		if root.label: return 0

		return max(self._height(root.left), self._height(root.right))+1

	def _leaves(self, root):

		if root.label: return 1

		return self._leaves(root.left) + self._leaves(root.right)

	def _validation_accuracy(self, validation_set, validation_labels):
		"""
		Returns the validation error of the tree
		"""

		acc = 0
		for i,instance in enumerate(validation_set):
			if self.predict(instance) == validation_labels[i]: acc +=1
		return acc*1.0/validation_set.shape[0]


	def prune(self, validation_set, validation_labels):
		"""
		Prunes the tree
		"""
		queue = deque([self.root])
		acc_func = []
		while(len(queue)):
			# Till Queue is not Empty

			root = queue.popleft()
			# Check Initial Error
			init_val_acc = self._validation_accuracy(validation_set, validation_labels)
			# Assume the node becomes a leaf
			root.label = 1 if root.positive_instances >= root.negative_instances else -1
			# Check final error
			prune_val_acc = self._validation_accuracy(validation_set,validation_labels)

			# See if changes are valid
			if(prune_val_acc < init_val_acc): root.label = None
			else:
				acc_func.append(prune_val_acc)
				continue

			# Continue for its leaves
			if root.left.label == None: queue.append(root.left)
			if root.right.label == None: queue.append(root.right)

		
		self.height = self._height(self.root)
		self.leaves = self._leaves(self.root)
		return acc_func


	def accuracy(self, feature_matrix, labels):
		"""
		Returns the accuracy of the tree on a feature matrix
		"""

		accuracy = 0
		for i,instance in enumerate(feature_matrix):
			if self.predict(instance) == labels[i]:
				accuracy += 1

		return accuracy*1.0/labels.shape[0]









	