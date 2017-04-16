from __future__ import print_function

from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell
from train import *

print(tf.__version__)

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
        matrix += [getPickVector(pick, isNextPick)]
    return matrix


# pass a batch of games that are all the same size
def getBatchTensor(games):
    return [getGameMatrix(game) for game in games]


# create LSTM constructor
lstm = tf.contrib.rnn.LSTMCell(M, state_is_tuple=False)

# cells = MultiRNNCell([lstm]*19)

# create tensor to store embedded picks per state per batch
X = tf.placeholder(tf.float32, [BATCH_SIZE, 20, N + 3], name="X")

W_1 = tf.Variable(tf.random_uniform([N + 3, M], -1.0, 1.0), name='W1')
b_1 = tf.Variable(tf.zeros([1, M]), name='b1')


# T is 3D and M is 2D. returns a 3D
def tensormult(T, M):
    i, j, k = T.get_shape()
    k0, l = M.get_shape()
    assert k == k0
    tensor = []
    for a in range(i):
        tensor.append(tf.matmul(T[a,:,:], M))
    return tf.stack(tensor)

X_ = tf.sigmoid(tf.add(tensormult(X, W_1), b_1), name="Xprime")

print(X.get_shape())
print(tf.tensordot(X, W_1, axes=1).get_shape())
print(X_.get_shape())

# X_ = tf.placeholder(tf.float32, [None, 20, N+3], name="X'")
Y = []

W_2 = tf.Variable(tf.random_uniform([M, N], -1.0, 1.0), name='W2')
b_2 = tf.Variable(tf.zeros([1, N]), name='b2')

initial_state = state = tf.zeros([M, M])

# output, state = tf.nn.dynamic_rnn(cells, X_, dtype=tf.float32)

# build graph and loss function expression
loss = 0.0
with tf.variable_scope("lstm") as scope:
    for i in range(19):
        if i > 0:
            scope.reuse_variables()

        output, state = lstm(tf.transpose(X_[:, i, :]), tf.transpose(state))

        y0 = tf.add(tf.matmul(output, W_2), b_2)
        y = tf.nn.softmax(y0, name='Output2-normalized')

        loss += tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(y0), labels=X_[:, i + 1, :],
                                                        name='cross_entropy' + str(i))
        Y.append(y)

final_state = state

if __name__ == "__main__":
    with tf.Session() as session:
        print("reading training data from", args.train + "..")
        train = [game for game in json.load(open(args.train, "r")) if len(game["picks_bans"]) == 20]
        print("read training data")

        # these things are now numbers, not graph nodes
        num_state = initial_state.eval()
        num_loss = 0.0

        for i in range(len(train)):
            batch = random.sample(train, BATCH_SIZE)
            batchTensor = getBatchTensor(batch)

            num_state, current_loss = session.run([final_state, loss],
                                                  feed_dict={initial_state: num_state, X: batchTensor})
            num_loss += current_loss
