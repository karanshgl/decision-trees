import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 2:
	print("Enter the CSV file as an argument")
	exit()

data = pd.read_csv(sys.argv[1], delimiter = ',', header = None)
# data2 = pd.read_csv(sys.argv[2], delimiter = ',', header = None)
# data2 = np.array(data2)
# data3 = pd.read_csv(sys.argv[3], delimiter = ',', header = None)
# data3 = np.array(data3)
data = np.loadtxt(sys.argv[1], delimiter = ',')
data = np.array(data)
fig, ax = plt.subplots()

plt.title("Noise")
plt.xlabel("Number of Nodes Flipped")
plt.ylabel("Accuracy")
ax.plot(data[:,0], data[:,1], color = 'r', label = 'Test Accuracy')
# ax.plot(data[:,0], data[:,2], color = 'b', label = 'Test Accuracy')
# ax.plot(data3[:,0], data3[:,1], color = 'g', label = 'Test Accuracy')
# legend = plt.legend(loc='center right')
plt.show()