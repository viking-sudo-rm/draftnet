import pickle, random
import tensorflow as tf

N = 113

flatten = lambda l: [item for sublist in l for item in sublist]

def getOneHot(pick):
	return [1 if i == pick["hero_id"] - 1 else 0 for i in range(N)]

def getData(filename):
	raw = pickle.load(open(filename, "rb"))
	X = [flatten(map(getOneHot, game["picks_bans"][:-1])) + [game["radiant_win"]] for game in raw]
	Y = [getOneHot(game["picks_bans"][-1]) for game in raw]
	return X, Y


print "reading training data.."
X, Y = getData("data-100000/train-100000.data")

x = tf.placeholder(tf.float32, shape=[1, len(X[0])])
# need to shape [batch_size, 1] for nn.nce_loss
y = tf.placeholder(tf.float32, shape=[1, N])

W = tf.Variable(
    tf.random_uniform([len(X[0]), N],-1.0, 1.0))
b = tf.Variable(tf.zeros([1, N]))

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = tf.matmul(x, W) + b, labels = y)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

trials = zip(X, Y)
print range(1000) == None
for step in range(1000):
  batch_xs, batch_ys = zip(*random.sample(trials, 1))
  _, loss_val = sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
  if step % 10 == 0:
  	print "Loss at ", step, loss_val # Report the loss

print "reading test data.."
X_, Y_ = getData("data-100000/test-100000.data")

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, feed_dict={x: X_, y: Y_})


