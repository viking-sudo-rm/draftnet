from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse, json, random
from display import *

N = 113     # number of heroes
M = 25
LEARNING_RATE = 0.001

BATCH_SIZE = 100
NUM_BATCHES = 1000000 # this number controls how long the program trains
EPOCHS = 100

# takes very close to 2^n iterations to reduce loss by one place with NUM_BATCHES = 1000000

PICK_THRESHOLD = 0.9

# create our nodes for the graph.

X = tf.placeholder(dtype=tf.float32, shape=[None, 4*N+1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, N])

# define the matrices (weights)

W_1 = tf.Variable(tf.random_uniform([4*N+1, M], -1.0, 1.0), name='W1')
b_1 = tf.Variable(tf.zeros([1, M]), name='b1')
W_2 = tf.Variable(tf.random_uniform([M, N], -1.0, 1.0), name='W2')
b_2 = tf.Variable(tf.zeros([1, N]), name='b2')

# define operations

h = tf.sigmoid(tf.add(tf.matmul(X, W_1), b_1))     # creates the embedded game information.
y0 = tf.add(tf.matmul(h, W_2), b_2)
Y_ = tf.sigmoid(y0)                                # probability distribution of next hero.

# define loss function:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=Y, name='cross_entropy')
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# load games

argparser = argparse.ArgumentParser(description="Set train and test files.")
argparser.add_argument('--train', help='path to train file', default='data/train-36740.json')
argparser.add_argument('--test', help='path to test file', default='data/test-5000.json')
args = argparser.parse_args()

# create function whose input is a game and output is a list of pairs (the data in the correct format)

flatten = lambda l: [item for sublist in l for item in sublist]

def getOneHot(pick):
    return [1 if i == getShiftedID(pick["hero_id"]) else 0 for i in range(N)]

def format(game):
    picks_bans = game['picks_bans']
    first_pick = picks_bans[0]['team']      # gives a 0 or a 1
    output = []
    team0picks = [0]*N
    team0bans = [0]*N
    team1picks = [0]*N
    team1bans = [0]*N
    if first_pick == 0:
        team0bans = getOneHot(picks_bans[0])
    else:
        team1bans = getOneHot(picks_bans[0])
    for i in range(1, 20):
        hero_id = getShiftedID(picks_bans[i]['hero_id'])
        is_pick_bit = 1 if picks_bans[i]['is_pick'] else 0
        if picks_bans[i]['team'] == 0:
            a = team0picks + team0bans + team1picks + team1bans + [is_pick_bit]
            output.append((a, getOneHot(picks_bans[i])))
            if picks_bans[i]['is_pick']:
                team0picks[hero_id] = 1 # this is way faster
                # team0picks = list(map(add, team0picks, getOneHot(picks_bans[i])))
            else:
                team0bans[hero_id] = 1
                # team0bans = list(map(add, team0bans, getOneHot(picks_bans[i])))
        else:
            a = team1picks + team1bans + team0picks + team0bans + [is_pick_bit]
            output.append((a, getOneHot(picks_bans[i])))
            if picks_bans[i]['is_pick']:
                team1picks[hero_id] = 1
                # team1picks = list(map(add, team1picks, getOneHot(picks_bans[i])))
            else:
                team1bans[hero_id] = 1
                # team1bans = list(map(add, team1bans, getOneHot(picks_bans[i])))
    return output

# extracts a set of indexes of likely picks from a probability distribution over heroes
# distribution is a numpy array representing a probability distribution
# notAllowed is a list of binary flags for heroes not allowable in the current context
def getPicks(distribution, notAllowed):

    # found the max-allowed prediction
    max_p = 0.0
    for i, p in enumerate(distribution):
        if p > max_p and not notAllowed[i]:
            max_p = p
    # max_p = np.argmax(distribution)

    # find other heroes over PICK_THRESHOLD
    picks = set()
    for i, p in enumerate(distribution):
        if p > max_p * PICK_THRESHOLD and not notAllowed[i]:
            picks.add(i)
    return picks

# collapse a context list x into a flag-set of non-allowable heroes
def getNotAllowed(context):
    notAllowed = [False] * N
    for i in range(len(context) - 1): #whenever we add more extra bits, need to change this number
        if context[i] == 1: notAllowed[i % N] = True
    return notAllowed

if __name__ == "__main__":
    with tf.Session() as session:

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        print("reading training data..")
        train = [game for game in json.load(open(args.train, "r")) if len(game["picks_bans"]) == 20]
        print("building trials..")
        trials = flatten([format(game) for game in train])

        print("starting training..")
        for i in range(EPOCHS):

            epochLoss = 0.0

            for _ in range(NUM_BATCHES // EPOCHS):
                x, y = zip(*random.sample(trials, BATCH_SIZE))
                epochLoss += session.run([cross_entropy, train_step], feed_dict={X: x, Y: y})[0]

            # increasing batch size increases error -- perhaps we should adjust something in the optimization

            print("epoch", i, "loss:", '{:.2f}'.format(sum(epochLoss)))

        save_path = saver.save(session, "results/bag-{}-{}-{}-{}.ckpt".format(BATCH_SIZE, NUM_BATCHES, LEARNING_RATE, M))
        print("saved session to", save_path)

        print("reading testing data..")
        test = [game for game in json.load(open(args.test, "r")) if len(game["picks_bans"]) == 20]

        print("starting testing..")
        counts = [0.0] * 19
        neighborhood_sizes = [0.0] * 19
        for game in test:
            x, y = zip(*format(game))
            distributions = session.run(Y_, feed_dict={X: x, Y: y})
            #TODO change this to be 
            for i, distribution in enumerate(distributions):
                notAllowed = getNotAllowed(x[i])
                picks = getPicks(distribution, notAllowed)
                neighborhood_sizes[i] += len(picks)
                if np.argmax(y[i]) in picks:
                    counts[i] += 1
        print("accuracies:", [c / len(test) for c in counts])
        print("neighborhood sizes:", [n / len(test) for n in neighborhood_sizes])


