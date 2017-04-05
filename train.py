import pickle, random
import tensorflow as tf
import numpy as np
from display import *

N = 113
M = 2 * N
L = 5 + N

BATCH_SIZE = 10
LEARNING_RATE = 0.5

flatten = lambda l: [item for sublist in l for item in sublist]

def getOneHot(pick):
	return [1 if i == pick["hero_id"] - 1 else 0 for i in range(N)]

def getData(filename, winOnly = True):
	raw = pickle.load(open(filename, "rb"))
	won = lambda game: (game["picks_bans"][-1]["team"] == 0 and game["radiant_win"]) or (game["picks_bans"][-1]["team"] == 1 and not game["radiant_win"])
	X = [(flatten(map(getOneHot, game["picks_bans"][:-1])) + [game["picks_bans"][0]["team"]] + [] if winOnly else [won(game)], getOneHot(game["picks_bans"][-1])) for game in raw if len(game["picks_bans"]) == 20 and (not winOnly or won(game))]
	return X

print "reading training data.."
trials = getData("data-41705/train-40705.data")

# None allows us to feed in variable-length inputs
x = tf.placeholder(tf.float32, shape=[None, len(trials[0][0])])
y_ = tf.placeholder(tf.float32, shape=[None, N])

W_1 = tf.Variable(
    tf.random_uniform([len(trials[0][0]), M],-1.0, 1.0))
b_1 = tf.Variable(tf.zeros([1, M]))

h_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_1), b_1))

W_2 = tf.Variable(
    tf.random_uniform([M, N],-1.0, 1.0))
b_2 = tf.Variable(tf.zeros([1, N]))

# h_2 = tf.nn.sigmoid(tf.add(tf.matmul(h_1, W_2), b_2))

# W_3 = tf.Variable(
#     tf.random_uniform([L, N],-1.0, 1.0))
# b_3 = tf.Variable(tf.zeros([1, N]))

y0 = tf.matmul(h_1, W_2) + b_2
y = tf.nn.softmax(y0)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y0, labels = y_)
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	print "starting training.."
	sess.run(init)
	for step in range(len(trials) / BATCH_SIZE):
	  batch_xs, batch_ys = zip(*random.sample(trials, BATCH_SIZE))
	  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	 #TODO: train batches
	 #TODO: more layers
	 #TODO: try backwards architecture

	print "reading test data.."
	tests = getData("data-41705/test-1000.data")

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sum = 0
	for testx, testy in tests:

		values = sess.run(y, feed_dict={x: [testx], y_: [testy]})
		predicted = np.argmax(values, 1)
		actual = np.argmax(testy, 0)

		if actual == predicted[0]: sum += 1

		# print predicted, actual, values[0][actual]

		h = []
		for i in range(0, len(testx) - N, N):
			h.append(vec2hero(testx[i:i + N])),

		teams = [Team("picking team"), Team("other team")]
		for i in range(len(PICK_BAN_ORDER)):
			team = teams[PICK_BAN_ORDER[i][1]]
			if PICK_BAN_ORDER[i][0]:
				team.pick(h[i])
			else:
				team.ban(h[i])

		print teams[0]
		print teams[1]

		print "predicted:", getName(int2hero(predicted))
		print "actual:", getName(int2hero(actual))
		print "-----------------------------"

	print sum, "/", len(tests)

	X, Y = zip(*tests)
	print(sess.run(accuracy, feed_dict={x: X, y_: Y}))