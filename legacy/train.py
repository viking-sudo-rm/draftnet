from __future__ import print_function
import json, random
import tensorflow as tf
import numpy as np
from display import *
import argparse
from sys import version_info

N = 113  # number of heroes
M = 25

BATCH_SIZE = 50
LEARNING_RATE = 0.0001  # should use something like 0.0001 for multi-layered network

argparser = argparse.ArgumentParser(description="Set train and test files.")

argparser.add_argument('--train', help='path to train file', default='data/train-8100.json')
argparser.add_argument('--test', help='path to test file', default='data/test-900.json')
argparser.add_argument('--model', help='path to model file', default='results/model-100-10000-0.0001-25.ckpt')

args = argparser.parse_args()

# the following three functions are to construct our vectors from the file.
flatten = lambda l: [item for sublist in l for item in sublist]

ROLES = ['Support', 'Jungler', 'Escape', 'Carry', 'Durable', 'Nuker', 'Pusher', 'Disabler', 'Initiator']


# need these functions to accomodate different versions of tensorflow
def getConcatNew(l, axis):
    return tf.concat(l, axis=axis)


def getConcatOld(l, axis):
    return tf.concat(axis, l)


def getConcat(l, axis):
    return getConcatOld(l, axis) if 2 in version_info else getConcatNew(l, axis)


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



# pickle.dump(p, open("p.data", "wb"), protocol=2)
def getData(filename, winOnly=True, xFeatures=False, yFeatures=False):
    # raw is a list with all the games. each game is a dict.
    raw = json.load(open(filename, "r"))
    # won is a bool. true only when the team with the last pick won.
    won = lambda game: (game["picks_bans"][-1]["team"] == 0 and game["radiant_win"]) or (
        game["picks_bans"][-1]["team"] == 1 and not game["radiant_win"])
    X = [([heroVectors(game, xFeatures)[i] for i in range(19)], heroVectors(game, xFeatures)[19], game["match_id"])
         for game in raw if won(game) and len(game["picks_bans"]) == 20]
    return X


if __name__ == "__main__":

    print("reading training data from", args.train + "..")
    trials = getData(args.train)

    print("setting up network..")

    # we create placeholders where our data will go. None allows us to feed in variable-length inputs.
    # x is now the placeholder for a single hero vector
    # x = tf.placeholder(tf.float32, shape=[None, len(trials[0][0][0])])
    # x = tf.placeholder(tf.float32, shape=[None, len(trials[0][0]), len(trials[0][0][0])])

    x = [tf.placeholder(tf.float32, shape=[None, N]) for i in range(19)]
    y_ = tf.placeholder(tf.float32, shape=[None, len(trials[0][1])])

    # we create the set of initial weights and biases.
    # the dimensions are so that (x*W_1 + b_1 )*W_2+b_2 has the same shape as y_
    W_1 = tf.Variable(tf.random_uniform([len(trials[0][0][0]), M], -1.0, 1.0), name='Weights1')
    b_1 = tf.Variable(tf.zeros([1, M]), name='Bias1')

    W_2 = tf.Variable(tf.random_uniform([M * 19, len(trials[0][1])], -1.0, 1.0), name='Weights2')
    b_2 = tf.Variable(tf.zeros([1, len(trials[0][1])]), name='Bias2')


    # If we add a third layer of length M
    # W_2 = tf.Variable(tf.random_uniform([M*19, M], -1.0, 1.0), name='Weights2')
    # b_2 = tf.Variable(tf.zeros([1, M]), name='Bias2')
    # W_3 = tf.Variable(tf.random_uniform([M, len(trials[0][1])], -1.0, 1.0), name='Weights3')
    # b_3 = tf.Variable(tf.zeros([1, len(trials[0][1])]), name='Bias3')

    def lower_dim(z, matrix, bias):
        out = [None] * 19
        for i in range(19):
            out[i] = tf.add(tf.matmul([z[i]], matrix), bias)
        w = tf.concat([out[i] for i in range(19)], 1)
        return w[0]


    # we create a model without an M hidden layer.
    def model2(W_1, b_1, W_2, b_2):
        x_prime = [tf.add(tf.matmul(x[i], W_1), b_1) for i in range(len(x))]
        h_1 = tf.sigmoid(getConcat(x_prime, 1), name="hidden1")  # axis=1; order of parameters different across versions

        y0 = tf.add(tf.matmul(h_1, W_2), b_2)
        y = tf.nn.softmax(y0, name='Output2-normalized')
        return y0, y


    # we create a model with an M hidden layer.
    def model3(W_1, b_1, W_2, b_2, W_3, b_3):
        x_prime = [tf.add(tf.matmul(x[i], W_1), b_1) for i in range(len(x))]
        h_1 = tf.sigmoid(getConcat(x_prime, 1), name="hidden1")  # axis=1; order of parameters different across versions

        h_2 = tf.sigmoid(tf.add(tf.matmul(h_1, W_2), b_2))
        y0 = tf.add(tf.matmul(h_2, W_3), b_3)
        y = tf.nn.softmax(y0, name='Output3-normalized')
        return y0, y


    y0, y = model2(W_1, b_1, W_2, b_2)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=y_, name='cross_entropy')
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # create a graph that can be viewed on tensorboard.
        writer = tf.summary.FileWriter(logdir="/tmp/tf2",
                                       graph=sess.graph)  # run python -m tensorflow.tensorboard --logdir=
        merged = tf.summary.merge_all(key='summaries')

        # initialize the variables
        sess.run(init)

        # start the training.
        # since we have len(trials) many test points, and we are feeding it
        # BATCH_SIZE many points at the time we will update our weights and biases a total of len(trials)/BATCH_SIZE
        print("starting training..")

        for step in range(len(trials)):  # len(trials) // BATCH_SIZE
            batch_xs, batch_ys, batch_ids = zip(*random.sample(trials, BATCH_SIZE))
            extracted_batch_xs = list(zip(*batch_xs))
            graph_args = {x[i]: extracted_batch_xs[i] for i in range(len(extracted_batch_xs))}
            graph_args.update({y_: batch_ys})
            sess.run(train_step, feed_dict=graph_args)


        print("reading test data from", args.test + "..")
        tests = getData(args.test)

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

            NotAllowed = {h[i]['id'] for i in range(19)}
            actual = np.argmax(testy, 0)

            while True:
                predicted = np.argmax(values, 1)[0]
                if predicted in NotAllowed:
                    values[0][predicted] = 0
                else:
                    break

            neighborhood = [i for i in range(len(values[0])) if
                            abs(values[0][i] - values[0][predicted]) / values[0][predicted] < .6 and values[0][
                                i] not in NotAllowed]
            print(len(neighborhood))
            if actual in neighborhood: s += 1

            print("predicted:", getName(int2hero(predicted)))
            print("predicted neighborhood:", list(map(getName, list(map(int2hero, neighborhood)))))
            print("actual:", getName(int2hero(actual)))
            print("-----------------------------")

        print(s, "/", len(tests))

        print("saving embedding weights as CSV..")
        np.savetxt("results/W.csv", sess.run(W_1), delimiter=",")
