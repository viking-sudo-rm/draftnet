from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse, json, random
from util import *

M = 50 #TODO try changing M and see if it improves results?
LEARNING_RATE = 0.01

BATCH_SIZE = 100
NUM_BATCHES = 100000 # this number controls how long the program trains
EPOCHS = 100

# takes very close to 2^n iterations to reduce loss by one place with NUM_BATCHES = 1000000 and LEARNING_RATE = 0.0001

PICK_THRESHOLD = 0.1 #0.35

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
Y_ = tf.nn.softmax(y0)                                # probability distribution of next hero.

# define loss function:

# TODO can simplify this part since we know one distribution is one-hot
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=Y, name='cross_entropy')
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# need this to avoid issues when importing something using argparser
def parseDraftnetArgs():
    argparser = argparse.ArgumentParser(description="Set train and test files.")
    argparser.add_argument('--train', help='path to train file', default='data/train-36740.json')
    argparser.add_argument('--test', help='path to test file', default='data/test-5000.json')
    argparser.add_argument('--save', help='path to save model', default="results/bag-{}-{}-{}-{}.ckpt".format(BATCH_SIZE, NUM_BATCHES, LEARNING_RATE, M))
    argparser.add_argument('--model', help='path to model file', default=None)
    # argparser.add_argument('--threshold', help='thresold for deciding pick set membership', default=0)
    return argparser.parse_args()

flatten = lambda l: [item for sublist in l for item in sublist]

# create function whose input is a game and output is a list of pairs (the data in the correct format)
def getOneHot(pick):
    return [1 if i == getShiftedID(pick["hero_id"]) else 0 for i in range(N)]

# extract 10 winning picks from a game
# TODO: should also have one that returns everything(sorted by radiant-dire) and predicts win
def format(game):
    picks_bans = game['picks_bans']
    winning_team = 0 if game['radiant_win'] else 1
    output = []

    team0, team1 = Team(), Team() # team 0 is the winning/picking team

    for i in range(20):
        hero_id = getShiftedID(picks_bans[i]['hero_id'])
        is_pick_bit = 1 if picks_bans[i]['is_pick'] else 0

        if picks_bans[i]['team'] == winning_team:
            a = getContext(team0, team1, is_pick_bit, winning_team)
            output.append((a, getOneHot(picks_bans[i])))
            if picks_bans[i]['is_pick']:
                team0.pick(APIHero.byID(hero_id))
            else:
                team0.ban(APIHero.byID(hero_id))
        else: # don't append to output, but pick the hero
            if picks_bans[i]['is_pick']:
                team1.pick(APIHero.byID(hero_id))
            else:
                team1.ban(APIHero.byID(hero_id))

    return output

def getNames(picks):
    return [APIHero.byID(pick).getName() for pick in picks]

def getContext(team0, team1, isPick, side):
    return team0.getContextVector() + team1.getContextVector() + [1 if isPick else 0] + [side]

def getDistribution(context, session):
    return session.run(Y_, feed_dict={X: [context]})[0]

# collapse a context list x into a flag-set of non-allowable heroes
def getNotAllowed(context):
    notAllowed = [False] * N
    for i in range(len(context) - (len(context) % N)): #whenever we add more extra bits, need to change this number
        if context[i] == 1: notAllowed[i % N] = True
    return notAllowed

# returns a list of possible picks from greatest to least probability
def getSuggestions(distribution, notAllowed, PICK_THRESHOLD=PICK_THRESHOLD):

    # found the max-allowed prediction
    max_p = 0.0
    for i, p in enumerate(distribution):
        if p > max_p and not notAllowed[i]:
            max_p = p
    # max_p = np.argmax(distribution)

    # find other heroes over PICK_THRESHOLD
    picks = []
    for i, p in enumerate(distribution):
        if p > max_p * PICK_THRESHOLD and not notAllowed[i]:
            picks.append(i)

    # return the picks from greatest to least probability
    return sorted(picks, key=lambda i: 1 - distribution[i])

def testInSession(test, session, PICK_THRESHOLD=PICK_THRESHOLD):
    print("starting testing with PICK_THRESHOLD={}..".format(PICK_THRESHOLD))
    counts = [0.0] * 10
    neighborhood_sizes = [0.0] * 10
    for game in test:
        x, y = zip(*format(game))
        distributions = session.run(Y_, feed_dict={X: x, Y: y})
        #TODO change this to be 
        for i, distribution in enumerate(distributions):
            notAllowed = getNotAllowed(x[i])
            picks = getSuggestions(distribution, notAllowed, PICK_THRESHOLD=PICK_THRESHOLD)
            neighborhood_sizes[i] += len(picks)
            if np.argmax(y[i]) in picks:
                counts[i] += 1
    print("accuracies:", [c / len(test) for c in counts])
    print("neighborhood sizes:", [n / len(test) for n in neighborhood_sizes])


if __name__ == "__main__":

    args = parseDraftnetArgs()

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if not args.model:

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

            save_path = saver.save(session, args.save)
            print("saved session to", save_path)

        else:
            saver.restore(session, args.model)

        print("reading testing data..")
        test = [game for game in json.load(open(args.test, "r")) if len(game["picks_bans"]) == 20]
        testInSession(test, session)

else:

    session = tf.Session() # session for others to use
