TRAIN_BOW = './aclImdb/train/labeledBow.feat'
TEST_BOW = './aclImdb/test/labeledBow.feat'

from random import shuffle
from word2bin import word2bin
import numpy as np
import pandas as pd


def get_sample(sample_size, keep_validation):
	"""
	Returns a training and validation set with sample size number of positive and negative examples
	"""
	validation_size = int(sample_size*keep_validation)

	train_space = []
	with open(TRAIN_BOW) as fp:
		train_space = fp.readlines()
	train_positive_space = [[1] + sample.split(' ')[1:] for sample in train_space if int(sample.split(' ')[0]) >=7]
	train_negative_space = [[-1] + sample.split(' ')[1:] for sample in train_space if int(sample.split(' ')[0]) <=4]

	# Positive
	pos_indices = list(range(0,len(train_positive_space)))
	shuffle(pos_indices)
	train_positive_sample = [train_positive_space[i] for i in pos_indices[:sample_size]]
	validation_positive_sample = [train_positive_space[i] for i in pos_indices[sample_size:sample_size+validation_size]]

	# Write in a file
	with open('./sample/positive_train_indices.txt', 'w') as fp:
		for line in pos_indices[:sample_size]:
			fp.write(str(line)+'\n')

	with open('./sample/positive_validation_indices.txt', 'w') as fp:
		for line in pos_indices[sample_size:sample_size+validation_size]:
			fp.write(str(line)+'\n')


	# Negative
	neg_indices = list(range(0,len(train_negative_space)))
	shuffle(neg_indices)
	train_negative_sample = [train_negative_space[i] for i in neg_indices[:sample_size]]
	validation_negative_sample = [train_negative_space[i] for i in neg_indices[sample_size:sample_size+validation_size]]

	# Write in a file
	with open('./sample/negative_train_indices.txt', 'w') as fp:
		for line in neg_indices[:sample_size]:
			fp.write(str(line)+'\n')

	with open('./sample/negative_validation_indices.txt', 'w') as fp:
		for line in neg_indices[sample_size:sample_size+validation_size]:
			fp.write(str(line)+'\n')

	training_set = train_positive_sample + train_negative_sample
	shuffle(training_set)

	validation_set = validation_positive_sample + validation_negative_sample
	shuffle(validation_set)

	return training_set, validation_set

def get_test(sample_size):
	"""
	Returns Test Set with sample size number of positive and negative examples
	"""
	test_space = []
	with open(TEST_BOW) as fp:
		test_space = fp.readlines()
	test_positive_space = [[1] + sample.split(' ')[1:] for sample in test_space if int(sample.split(' ')[0]) >=7]
	test_negative_space = [[-1] + sample.split(' ')[1:] for sample in test_space if int(sample.split(' ')[0]) <=4]

	# Positive
	pos_indices = list(range(0,len(test_positive_space)))
	shuffle(pos_indices)
	test_positive_sample = [test_positive_space[i] for i in pos_indices[:sample_size]]

	# Write in a file
	with open('./sample/positive_test_indices.txt', 'w') as fp:
		for line in pos_indices[:sample_size]:
			fp.write(str(line)+'\n')


	# Negative
	neg_indices = list(range(0,len(test_negative_space)))
	shuffle(neg_indices)
	test_negative_sample = [test_negative_space[i] for i in neg_indices[:sample_size]]

	# Write in a file
	with open('./sample/negative_test_indices.txt', 'w') as fp:
		for line in neg_indices[:sample_size]:
			fp.write(str(line)+'\n')


	test_set = test_positive_sample + test_negative_sample
	shuffle(test_set)

	return test_set



def get_data(dataset, clustering = True):
	"""
	Returns the instance matrix and corresponding labels of a training set

	Parameters:
	dataset: the result from get_sample method
	"""

	if clustering:
		w2b, bins = word2bin()

		matrix = np.zeros((len(dataset), len(bins)))
		labels = np.array([i[0] for i in dataset])

		for i,instance in enumerate(dataset):
			for features in instance[1:]:
				feature = list(map(int,features.split(':')))
				if feature[0] not in w2b.keys(): continue
				b_index = w2b[feature[0]]
				matrix[i,b_index] += feature[1]
	else:
		# Get most common words from vocab with decent polarity
		p_vals = []
		with open('./aclImdb/imdbEr.txt') as fp:
			p_vals = list(map(float, fp.readlines()))

		p_vals = [[i, val] for i,val in enumerate(p_vals)]
		p_vals.sort(key = lambda x: x[1])
		
		

	return matrix, labels

def to_csv(filename, data):
	df = pd.DataFrame(data)
	df.to_csv(filename, header=None, index = False)

def save_train_set(sample_size, keep_validation = 0.2):
	"""
	Saves samples in .csv file 
	"""
	sample_size = sample_size//2 # Since Equal Number of Positive and Negative Samples
	instance_set, validation_set = get_sample(sample_size, keep_validation)
	train_matrix, train_labels = get_data(instance_set)
	validation_matrix, validation_labels = get_data(validation_set)


	# Save Train Set
	filename_matrix = './data/train.csv'
	filename_labels = './data/train_labels.csv'

	# Save to csv
	to_csv(filename_matrix, train_matrix)
	to_csv(filename_labels, train_labels)

	# Save Validation
	filename_matrix = './data/validation.csv'
	filename_labels = './data/validation_labels.csv'

	# Save to csv
	to_csv(filename_matrix, validation_matrix)
	to_csv(filename_labels, validation_labels)

def save_test_set(sample_size):

	sample_size = sample_size//2 
	instance_set = get_test(sample_size)
	test_matrix, test_labels = get_data(instance_set)

	# Save Test Set
	filename_matrix = './data/test.csv'
	filename_labels = './data/test_labels.csv'

	# Save to csv
	to_csv(filename_matrix, test_matrix)
	to_csv(filename_labels, test_labels)






