TRAIN_BOW = './aclImdb/train/labeledBow.feat'
TEST_BOW = './aclImdb/test/labeledBow.feat'

from random import shuffle
from word2bin import word2bin
import numpy as np
import pandas as pd


def get_sample(sample_size, train = True):
	"""
	Returns a training set with sample size number of positive and negative examples
	"""
	train_or_test = 'train' if train else 'test'
	train_space = []
	with open(TRAIN_BOW if train else TEST_BOW) as fp:
		train_space = fp.readlines()
	train_positive_space = [[1] + sample.split(' ')[1:] for sample in train_space if int(sample.split(' ')[0]) >=7]
	train_negative_space = [[-1] + sample.split(' ')[1:] for sample in train_space if int(sample.split(' ')[0]) <=4]

	# Positive
	pos_indices = list(range(0,len(train_positive_space)))
	shuffle(pos_indices)
	train_positive_sample = [train_positive_space[i] for i in pos_indices[:sample_size]]

	# Write in a file
	with open('./sample/positive_'+ train_or_test +'_indices.txt', 'w') as fp:
		for line in pos_indices[:sample_size]:
			fp.write(str(line)+'\n')

	# Negative
	neg_indices = list(range(0,len(train_negative_space)))
	shuffle(neg_indices)
	train_negative_sample = [train_negative_space[i] for i in neg_indices[:sample_size]]

	# Write in a file
	with open('./sample/negative_'+ train_or_test +'_indices.txt', 'w') as fp:
		for line in neg_indices[:sample_size]:
			fp.write(str(line)+'\n')

	training_set = train_positive_sample + train_negative_sample
	shuffle(training_set)

	return training_set


def get_train_set(training_set, clustering = True):
	"""
	Returns the instance matrix and corresponding labels of a training set

	Parameters:
	training_set: the result from get_sample method
	"""

	if clustering:
		w2b, bins = word2bin()

		train_matrix = np.zeros((len(training_set), len(bins)))
		train_labels = np.array([i[0] for i in training_set])

		for i,instance in enumerate(training_set):
			for features in instance[1:]:
				feature = list(map(int,features.split(':')))
				if feature[0] not in w2b.keys(): continue
				b_index = w2b[feature[0]]
				train_matrix[i,b_index] += feature[1]
	else:
		# Get most common words from vocab with decent polarity
		p_vals = []
		with open('./aclImdb/imdbEr.txt') as fp:
			p_vals = list(map(float, fp.readlines()))

		p_vals = [[i, val] for i,val in enumerate(p_vals)]
		p_vals.sort(key = lambda x: x[1])
		
		

	return train_matrix, train_labels



def save_random_set(sample_size, train = True):
	"""
	Saves samples in .csv file 
	"""
	instance_set = get_sample(sample_size, train)
	matrix, labels = get_train_set(instance_set)
	train_or_test = 'train' if train else 'test'

	filename_matrix = './data/'+ train_or_test +'.csv'
	filename_labels = './data/'+ train_or_test +'_labels.csv'

	# Save to csv
	df = pd.DataFrame(matrix)
	df.to_csv(filename_matrix, header=None, index = False)
	df = pd.DataFrame(labels)
	df.to_csv(filename_labels, header=None, index = False)

	return filename_matrix, filename_labels



