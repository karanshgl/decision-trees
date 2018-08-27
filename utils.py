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

def add_noise(labels, noise_percent = 0.005):
	"""
	Given the labels, it toggles the noise_perfect number of labels
	"""
	rows = int(labels.shape[0]*noise_percent)
	rows = np.random.choice(labels.shape[0], rows)
	labels[rows] = np.where(labels[rows] == 1, -1, 1)

	return labels











