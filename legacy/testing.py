import tensorflow as tf
from train import *

BATCH_SIZE = 1
LSTM_SIZE = 25
NUM_BATCHES = 10000
EPOCHS = 100

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
lstm = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, state_is_tuple=True)

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
        tensor.append(tf.matmul(T[a, :, :], M))
    return tf.stack(tensor)

X_ = tf.sigmoid(tf.add(tensormult(X, W_1), b_1), name="Xprime")

W_2 = tf.Variable(tf.random_uniform([M, N], -1.0, 1.0), name='W2')
b_2 = tf.Variable(tf.zeros([1, N]), name='b2')

# initial_state = state = tf.zeros([M, M])

# use dynamic_rnn to build the graph
Y, final_state = tf.nn.dynamic_rnn(lstm, X_, time_major=False, dtype=tf.float32)

predictedPicks = []

loss = 0.0
with tf.variable_scope("loss_layer") as scope:
    for i in range(19):
        if i > 0:
            scope.reuse_variables()
        y0 = tf.add(tf.matmul(Y[:, i, :], W_2), b_2)
        predictedPicks.append(tf.nn.softmax(y0, name='Output2-normalized'))
        loss += tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=X[:, i + 1, :113],
                                                        name='cross_entropy' + str(i))

loss = tf.reduce_sum(loss)
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

if __name__ == "__main__":
    with tf.Session() as session:
        saver = tf.train.Saver()
        save_path = saver.restore(session, args.model)
        print("Model loaded")
        print("reading test data from", args.test + "..")
        test = [game for game in json.load(open(args.test, "r")) if len(game["picks_bans"]) == 20]
        print("read test data")

        sums = [0.0 for _ in range(19)]

        allowed = [True for _ in range(N)]

        for game in test:
            actual = getGameMatrix(game)
            predictions_distribution = session.run(predictedPicks, feed_dict={X: [actual]}) #length 19

            print(game["match_id"])

            predictions = []

            for i in range(19):
                while True:
                    candidate = np.argmax(predictions_distribution[i])
                    # print(predictions_distribution[i])
                    if allowed[candidate]:
                        predictions.append(candidate)
                        break
                    else:
                        predictions_distribution[i] = 0
                allowed[actual[i].index(1)] = False
            print(predictions)
