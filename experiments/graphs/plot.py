import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

if len(sys.argv) != 2:
	print("Enter the CSV file as an argument")
	exit()

data = pd.read_csv(sys.argv[1], delimiter = ',', header = None)
data = np.array(data)

plt.plot(data[:,0], data[:,1])
plt.show()