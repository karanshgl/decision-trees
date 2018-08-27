import numpy as np
from collections import defaultdict
import pickle

def word2bin(bin_size = 0.01):
	"""
	Returns:
	dict: with word indices as keys and bin index as value
	list: bin index to bin value
	"""
	w2b = dict()
	p_vals = []
	with open('./aclImdb/imdbEr.txt') as fp:
		p_vals = list(map(float, fp.readlines()))

	# Remove the polarities between -0.5 and 0.5
	p_vals = [[i, val] for i,val in enumerate(p_vals) if val>1 or val<-(1)]
	# Remove duplicates and convert to a numpyarray
	p_vals = np.array(p_vals)
	# Create bins
	bins_positive = np.arange(1,4.5, bin_size)
	bins_negative = np.arange(-4.5,-1, bin_size)
	bins = np.append(bins_negative, bins_positive)
	
	# Perform bining, values are bin indices
	i2b = np.digitize(p_vals[:,1], bins, right = True)

	for i in range(len(i2b)):
		w2b[int(p_vals[i,0])] = (i2b[i]-1)

	# Store in a file
	with open('./bin/w2b.pkl','wb') as fp:
		pickle.dump(w2b, fp)
	with open('./bin/bins.pkl', 'wb') as fp:
		pickle.dump(bins,fp)

	return w2b, bins

