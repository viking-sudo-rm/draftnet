import pickle, random
import tensorflow as tf
import numpy as np

N = 113

flatten = lambda l: [item for sublist in l for item in sublist]

def getOneHot(pick):
	return [1 if i == pick["hero_id"] - 1 else 0 for i in range(N)]

def getData(filename, winOnly = True):
	raw = pickle.load(open(filename, "rb"))
	won = lambda game: (game["picks_bans"][-1]["team"] == 0 and game["radiant_win"]) or (game["picks_bans"][-1]["team"] == 1 and not game["radiant_win"])
	X = [(flatten(map(getOneHot, game["picks_bans"][:-1])) + [game["radiant_win"]], getOneHot(game["picks_bans"][-1])) for game in raw if len(game["picks_bans"]) == 20 and (!winOnly or won(game))]
	return X


print "reading training data.."
trials = getData("data-41705/train-40705.data")

x = tf.placeholder(tf.float32, shape=[1, len(trials[0][0])])
# need to shape [batch_size, 1] for nn.nce_loss
y_ = tf.placeholder(tf.float32, shape=[1, N])

W = tf.Variable(
    tf.random_uniform([len(trials[0][0]), N],-1.0, 1.0))
b = tf.Variable(tf.zeros([1, N]))

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = tf.matmul(x, W) + b, labels = y_)
y = tf.nn.softmax(tf.matmul(x, W) + b)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	print "starting training.."
	sess.run(init)
	for step in range(100000):
	  batch_xs, batch_ys = zip(*random.sample(trials, 1))
	  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	  # if step % 10 == 0:
	  # 	print "Loss at ", step, loss_val # Report the loss

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

		print predicted, actual, values[0][actual]

	print sum, "/", len(tests)

	# X, Y = zip(*tests)
	# print sess.run(correct_prediction, feed_dict={x: X, y_: Y})