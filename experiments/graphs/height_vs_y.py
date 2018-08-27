
xy_nodes = []
xy_acc = []

for i in range(51):
	tree = DecisionTree(max_height = i)
	tree.fit(train_set, train_labels)
	xy_nodes.append((int(i),tree.leaves))
	xy_acc.append((int(i),tree.accuracy(test_set, test_labels)))
	print("{} Done {}".format(i, tree.leaves))

xy_nodes = np.array(xy_nodes)
xy_acc = np.array(xy_acc)
np.savetxt("./experiments/graphs/height_vs_leaves.csv", xy_nodes, delimiter=",")
np.savetxt("./experiments/graphs/height_vs_accu.csv", xy_acc, delimiter=",")