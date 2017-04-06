from __future__ import print_function
import pickle, random
import tensorflow as tf
import numpy as np
from display import *


N = 113
M = N*10
L = 5 + N

BATCH_SIZE = 10
LEARNING_RATE = 0.5

# the following three functions are to construct our vectors from the file.
flatten = lambda l: [item for sublist in l for item in sublist]

ROLES = ['Support', 'Jungler', 'Escape', 'Carry', 'Durable', 'Nuker', 'Pusher', 'Disabler', 'Initiator']

# pick is a dict with keys: hero, team, order
def getOneHot(pick):
    return [1 if i == getShiftedID(pick["hero_id"]) else 0 for i in range(N)]

def getFeatures(pick):
    features = []
    hero = heroes[getShiftedID(pick["hero_id"])]
    if hero['attack_type'] == 'Melee':
        features.append(1)
    else:
        features.append(0)
    if hero['primary_attr'] == 'str':
        features += [1, 0, 0]
    elif hero['primary_attr'] == 'agi':
        features += [0, 1, 0]
    else:
        features += [0, 0, 1]
    features += [1 if ROLES[t] in hero['roles'] else 0 for t in range(9)]
    return features

def getData(filename, winOnly=True, xFeatures=True, yFeatures=False):
    # raw is a list with all the games. each game is a dict.
    raw = pickle.load(open(filename, "rb"))
    # won is a bool. true only when the team with the last pick won.
    won = lambda game: (game["picks_bans"][-1]["team"] == 0 and game["radiant_win"]) or (game["picks_bans"][-1]["team"] == 1 and not game["radiant_win"])

    X = [(flatten(map(lambda hero: getOneHot(hero) + (getFeatures(hero) if xFeatures else []), game["picks_bans"][:-1])) + [game["picks_bans"][0]["team"]] + [] if winOnly else [
        won(game)], getOneHot(game["picks_bans"][-1]) + (getFeatures(game["picks_bans"][-1]) if yFeatures else [])) for game in raw if
         len(game["picks_bans"]) == 20 and (not winOnly or won(game))]
    return X

print("reading training data..")
trials = getData("matches-100000.data")

# we create placeholders where our data will go. None allows us to feed in variable-length inputs.
x = tf.placeholder(tf.float32, shape=[None, len(trials[0][0])])
y_ = tf.placeholder(tf.float32, shape=[None, len(trials[0][1])])


# we create the set of initial weights and biases.
# the dimensions are so that (x*W_1 + b_1 )*W_2+b_2 has the same shape as y_
W_1 = tf.Variable(tf.random_uniform([len(trials[0][0]), M], -1.0, 1.0), name='Weights1')
b_1 = tf.Variable(tf.zeros([1, M]), name='Bias1')
W_2 = tf.Variable(tf.random_uniform([M, len(trials[0][1])], -1.0, 1.0), name='Weights2')
b_2 = tf.Variable(tf.zeros([1, len(trials[0][1])]), name='Bias2')

# we create our model.
def model(W_1, b_1, W_2,b_2):
    h_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W_1), b_1),name='Output1')
    y0 = tf.matmul(h_1, W_2) + b_2
    y = tf.nn.softmax(y0, name='Output2-normalized')
    return y0, y

y0, y = model(W_1, b_1, W_2, b_2)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=y_, name='cross_entropy')
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # create a graph that can be viewed on tensorboard.
    writer = tf.summary.FileWriter(logdir="/tmp/tf1", graph=sess.graph)  # run python -m tensorflow.tensorboard --logdir=
    merged = tf.summary.merge_all(key='summaries')

    #initialize the variables
    sess.run(init)

    # start the training.
    # since we have len(trials) many test points, and we are feeding it
    # BATCH_SIZE many points at the time we will update our weights and biases a total of len(trials)/BATCH_SIZE
    print("starting training..")
    for step in range(len(trials) // BATCH_SIZE):
        batch_xs, batch_ys = zip(*random.sample(trials, BATCH_SIZE))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # TODO: train batches
        # TODO: more layers
        # TODO: try backwards architecture

    print("reading test data..")

    tests = getData("matches-100000.data")

    # checks if the prediction is correct.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))    # its a boolean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast converts the bool into a float

    sum = 0
    for testx, testy in tests:

        values = sess.run(y, feed_dict={x: [testx], y_: [testy]})   # its a numpy.ndarray
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

        #print(teams[0])
        #print(teams[1])

        print("predicted:", getName(int2hero(predicted)))
        print("actual:", getName(int2hero(actual)))
        print("-----------------------------")

    print(sum, "/", len(tests))

    X, Y = zip(*tests)
    print(sess.run(accuracy, feed_dict={x: X, y_: Y}))
