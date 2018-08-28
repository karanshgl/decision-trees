Name: Karan Sehgal
Entry Number: 2016CSB1080

The project includes the following files:

# root directory
main.py: The main script file
preprocess.py: File contains the code that randomly selects instances and stores them as .csv file in the data directory
decision_tree.py: File containing the code for Decision Tree
utils.py: Some helper functions
word2bin.py: File contains the code for feature binning
random_forest.py: File contains the code for Random Forest

# experiments directory
exp_x.py, where x [2-5]: Contains the code for experiment x
## graphs directory: Contains datapoints, scripts and images for graphs

# data directory
selected-features-indices.txt: contains the indices of words used as features (clusted later into bins)
train/validation/test.csv: Feature Matrix for training/validation/test sets
train/validation/test_labels.csv: Labels for training/validation/test sets

# sample directory: contains the list of indices of selected samples

# bin directory: contains some binary files

# model directory: contains saved model files (pickle dumps)


### HOW TO RUN ###

to run, from root directory:
	$ python main.py X
	where X is the experiment number [2-5]

Note: In some experiments, you will be asked whether you wish to retrain the tree or not. Input should be either 0 or 1.