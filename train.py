from __future__ import print_function
import pickle, random
import tensorflow as tf
import numpy as np
from display import *
import argparse

N = 113 # number of heroes
M = 50

#ROLES = ['Support', 'Jungler', 'Escape', 'Carry', 'Durable', 'Nuker', 'Pusher', 'Disabler', 'Initiator']
#to disable/enable a feature, comment it out here

# Adding the following features seems to narrow predictions too much:
# Disabler
# Pusher
# Nuker
BATCH_SIZE = 50
LEARNING_RATE = 0.01 #should use something like 0.0001 for multi-layered network

argparser = argparse.ArgumentParser(description="Set train and test files.")

argparser.add_argument('--train', help='path to train file', default='data-41705/train-40705.data')
argparser.add_argument('--test', help='path to test file', default='data-41705/test-1000.data')

args = argparser.parse_args()

TRAIN_FILE = args.train
TEST_FILE = args.test

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
    features += [1 if ROLES[t] in hero['roles'] else 0 for t in range(len(ROLES))]
    return features

# returns a list of length 20. the ith entry is the vector corresponding to the ith pick
def heroVectors(game, Features):
    vector = [None] * 20
    for i in range(20):
        vector[i] = getOneHot(game["picks_bans"][i])
        if Features:
            vector[i] += getFeatures(game["picks_bans"])
    return vector

# TODO change getData to return correctly formatted batch data
# should also probably save as JSON or something instead of pickle
def getData(filename, winOnly=True, xFeatures=False, yFeatures=False):
    # raw is a list with all the games. each game is a dict.
    raw = pickle.load(open(filename, "rb"))
    # won is a bool. true only when the team with the last pick won.
    won = lambda game: (game["picks_bans"][-1]["team"] == 0 and game["radiant_win"]) or (
    game["picks_bans"][-1]["team"] == 1 and not game["radiant_win"])
    X = [([heroVectors(game, xFeatures)[i] for i in range(19)], heroVectors(game, xFeatures)[19], game["match_id"])
         for game in raw if won(game) and len(game["picks_bans"]) == 20]
    return X


print("reading training data..")
trials = getData(TRAIN_FILE)

# we create placeholders where our data will go. None allows us to feed in variable-length inputs.
# x is now the placeholder for a single hero vector
# x = tf.placeholder(tf.float32, shape=[None, len(trials[0][0][0])])
y_ = tf.placeholder(tf.float32, shape=[None, len(trials[0][1])])
#x = tf.placeholder(tf.float32, shape=[None, len(trials[0][0]), len(trials[0][0][0])])

x = [tf.placeholder(tf.float32, shape=[None, N]) for i in range(19)]

# we create the set of initial weights and biases.
# the dimensions are so that (x*W_1 + b_1 )*W_2+b_2 has the same shape as y_
W_1 = tf.Variable(tf.random_uniform([len(trials[0][0][0]), M], -1.0, 1.0), name='Weights1')
b_1 = tf.Variable(tf.zeros([1, M]), name='Bias1')

W_2 = tf.Variable(tf.random_uniform([M*19, len(trials[0][1])], -1.0, 1.0), name='Weights2')
b_2 = tf.Variable(tf.zeros([1, len(trials[0][1])]), name='Bias2')

#If we add a third layer of length M
# W_2 = tf.Variable(tf.random_uniform([M*19, M], -1.0, 1.0), name='Weights2')
# b_2 = tf.Variable(tf.zeros([1, M]), name='Bias2')
# W_3 = tf.Variable(tf.random_uniform([M, len(trials[0][1])], -1.0, 1.0), name='Weights3')
# b_3 = tf.Variable(tf.zeros([1, len(trials[0][1])]), name='Bias3')

def lower_dim(z, matrix, bias):
    out = [None]*19
    for i in range(19):
        out[i] = tf.add(tf.matmul([z[i]], matrix), bias)
    w = tf.concat([out[i] for i in range(19)], 1)
    return w[0]

# we create a model without an M hidden layer.
def model2(W_1, b_1, W_2, b_2):
    x_prime = [tf.add(tf.matmul(x[i],W_1), b_1) for i in range(len(x))]
    h_1 = tf.sigmoid(tf.concat(1, x_prime), name="hidden1") #axis=1; order of parameters different across versions
    
    y0 = tf.add(tf.matmul(h_1, W_2), b_2)
    y = tf.nn.softmax(y0, name='Output2-normalized')
    return y0, y

# we create a model with an M hidden layer.
def model3(W_1, b_1, W_2, b_2, W_3, b_3):
    x_prime = [tf.add(tf.matmul(x[i],W_1), b_1) for i in range(len(x))]
    h_1 = tf.sigmoid(tf.concat(1, x_prime), name="hidden1") #axis=1; order of parameters different across versions
    
    h_2 = tf.sigmoid(tf.add(tf.matmul(h_1, W_2), b_2))
    # TODO add another layer in here?
    y0 = tf.add(tf.matmul(h_2, W_3), b_3)
    y = tf.nn.softmax(y0, name='Output3-normalized')
    return y0, y


y0, y = model2(W_1, b_1, W_2, b_2)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=y_, name='cross_entropy')
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # create a graph that can be viewed on tensorboard.
    writer = tf.summary.FileWriter(logdir="/tmp/tf1",
                                   graph=sess.graph)  # run python -m tensorflow.tensorboard --logdir=
    merged = tf.summary.merge_all(key='summaries')

    # initialize the variables
    sess.run(init)

    # start the training.
    # since we have len(trials) many test points, and we are feeding it
    # BATCH_SIZE many points at the time we will update our weights and biases a total of len(trials)/BATCH_SIZE
    print("starting training..")
    training_bool = True
    for step in range(len(trials) // BATCH_SIZE):
        batch_xs, batch_ys, batch_ids = zip(*random.sample(trials, BATCH_SIZE))
        extracted_batch_xs = list(zip(*batch_xs))
        graph_args = {x[i] : extracted_batch_xs[i] for i in range(len(extracted_batch_xs))}
        graph_args.update({y_: batch_ys})
        sess.run(train_step, feed_dict=graph_args)
        # TODO: train batches
        # TODO: more layers
        # TODO: try backwards architecture

    print("reading test data..")
    tests = getData(TEST_FILE)

    # checks if the prediction is correct.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # its a boolean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # cast converts the bool into a float

    s = 0
    for testx, testy, testId in tests:
        graph_args = {x[i]: [testx[i]] for i in range(len(testx))}
        graph_args.update({y_: [testy]})
        values = sess.run(y, feed_dict=graph_args)
        h = []
        for v in testx:
            h.append(vec2hero(v))

        teams = [Team("picking team"), Team("other team")]
        for i in range(len(PICK_BAN_ORDER)):
            team = teams[PICK_BAN_ORDER[i][1]]
            if PICK_BAN_ORDER[i][0]:
                team.pick(h[i])
            else:
                team.ban(h[i])

        print("match:", testId)
        print(teams[0])
        print(teams[1])

        # print(values)
        # predicted = {values[0][i] : i for i in range(len(values[0]))}
        actual = np.argmax(testy, 0)
        predicted = np.argmax(values, 1)[0]

        # for i in sorted(predicted):
        #   heroId = int(predicted[i])
        #   hero = int2hero(heroId)
        #   if teams[0].isValid(hero) and teams[1].isValid(hero):
        #       break

        if actual == predicted: s += 1

        print("predicted:", getName(int2hero(predicted)))
        print("actual:", getName(int2hero(actual)))
        print("-----------------------------")


    print(s, "/", len(tests))
