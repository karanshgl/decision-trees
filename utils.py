import numpy as np

def entropy(p1,p2):
	"""
	Returns the entropy based on the parameters

	Parameters
	p1: probability of positive samples
	p2: probability of negative samples
	"""
	if p1 == 0 or p2 == 0: return 0
	return -(p1*np.log2(p1) + p2*np.log2(p2))


def best_split(feature, labels):
	"""
	Returns the best binary split value
	"""

	unique_values = np.unique(feature)
	# Each element of info contains a tuple of val, num_of_examples, label
	info = []
	total_left_elements = 0
	for value in unique_values:
		indices = np.argwhere(feature == value)
		majority = 1 if labels[indices[:,0]].mean()>=0 else -1
		total_left_elements += indices.shape[0]*majority
		info.append((value, total_left_elements))


	max_val = 0
	val_to_return = 0
	for element in info:
		if(abs(element[1]) > max_val):
			max_val = abs(element[1])
			val_to_return = element[0]

	return val_to_return








