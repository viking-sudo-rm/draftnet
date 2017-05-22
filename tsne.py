from __future__ import print_function
import sklearn.manifold, matplotlib
import matplotlib.pyplot as plt
from draftnet import *

args = parseDraftnetArgs()

with session.as_default():
	saver = tf.train.Saver()
	saver.restore(session, args.model)

	tsne = sklearn.manifold.TSNE(n_components=2, perplexity=30.0)

	print("fitting TSNE..")
	X_reduced = tsne.fit_transform(session.run(graph.W_2).transpose() if args.layer == 2 else None)
	# FIXME embeddings in the other direction unimplemented

	x, y = zip(*X_reduced) # separate into two lists

	fig, ax = plt.subplots()
	ax.scatter(x, y)

	# add text labels
	for i in range(len(X_reduced)):
		plt.annotate(
			APIHero.byID(i).getName(),
			xy=X_reduced[i],
			fontsize=6
		)

	plt.show()