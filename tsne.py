import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import display

X = np.loadtxt("results/W.csv", delimiter=",")

tsne = sklearn.manifold.TSNE(n_components=2, perplexity=30.0)

print "fitting TSNE.."
X_reduced = tsne.fit_transform(X)

x, y = zip(*X_reduced) # separate into two lists

fig, ax = plt.subplots()
ax.scatter(x, y)

# add text labels
for i in range(len(X)):
	plt.annotate(
		display.getName(display.int2hero(i)),
		xy=X_reduced[i],
		fontsize=6
	)

plt.show()