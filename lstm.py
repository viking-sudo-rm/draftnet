from __future__ import print_function
from train import *

isPick = lambda pick: 1 if pick["is_pick"] else 0
isRadiant = lambda pick: pick["team"]

def getPickVector(pick, isNextPick):
	return getOneHot(pick) + [isPick(pick), isRadiant(pick), isNextPick]

# this will only parse CM games
def getGameMatrix(game):
	matrix = []
	picks_bans = game["picks_bans"]
	for i in range(len(picks_bans)):
		pick = picks_bans[i]
		isNextPick = -1 if i == 19 else isPick(picks_bans[i + 1])
		matrix += getPickVector(pick, isNextPick)
	return matrix

# pass a batch of games that are all the same size
def getBatchTensor(games):
	return [getGameMatrix(game) for game in games]

# create LSTM constructor
lstm = tf.contrib.rnn_cell.BasicLSTMCell(M)

# create tensor to store embedded picks per state per batch
X = tf.placeholder(tf.float32, [None, 20, N + 3])

W_1 = tf.Variable(tf.random_uniform([M, N + 3], -1.0, 1.0), name='Weights1')
b_1 = tf.Variable(tf.zeros([1, N + 3]), name='Bias1')

X_ = tf.sigmoid(tf.add(tf.matmul(X, W_1), b_1))

Y = []

W_2 = tf.Variable(tf.random_uniform([M, N], -1.0, 1.0), name='Weights1')
b_2 = tf.Variable(tf.zeros([1, N]), name='Bias1')

initial_state = state = tf.zeros([None, lstm.state_size])

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=y_, name='cross_entropy')
loss = 0.0

# build graph and loss function expression
for i in range(19):

	output, state = lstm(X_[:, i, :], state)

	y0 = tf.add(tf.matmul(output, W_2), b_2)
	y = tf.nn.softmax(y0, name='Output2-normalized')
	
	loss += tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=X[:, i + 1, :], name='cross_entropy' + str(i))
	Y.append(y)

final_state = state

if __name__ == "__main__":

	print("reading training data from", args.train + "..")
	train = [game for game in json.load(open(args.train, "r")) if len(game["picks_bans"]) == 20]

	# these things are now numbers, not graph nodes
	num_state = initial_state.eval()
	num_loss = 0.0

	for i in range(len(trainGames)):

		batch = random.sample(train, BATCH_SIZE)
		batchTensor = getBatchTensor(batch)

		num_state, current_loss = session.run([final_state, loss],
			feed_dict={initial_state: num_state, X: game})
		num_loss += current_loss
