from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse, json, random, sys
from util import *

M = 50 #TODO try changing M and see if it improves results?
LEARNING_RATE = 0.01

PICK_THRESHOLD = 0.15 #0.35

# takes very close to 2^n iterations to reduce loss by one place with NUM_BATCHES = 1000000 and LEARNING_RATE = 0.0001

class DraftGraph(object):

    # we can give this class parameters so it's easy to construct different types of graphs with slight differences

    def __init__(self, logitsX, logitsY, logitsHidden=M):

        # create our nodes for the graph.

        self.X = tf.placeholder(dtype=tf.float32, shape=[None, logitsX])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, logitsY])

        # define the matrices (weights)

        self.W_1 = tf.Variable(tf.random_uniform([logitsX, logitsHidden], -1.0, 1.0), name='W1')
        self.b_1 = tf.Variable(tf.zeros([1, logitsHidden]), name='b1')
        self.W_2 = tf.Variable(tf.random_uniform([logitsHidden, logitsY], -1.0, 1.0), name='W2')
        self.b_2 = tf.Variable(tf.zeros([1, logitsY]), name='b2')

        # define operations

        self.h = tf.sigmoid(tf.add(tf.matmul(self.X, self.W_1), self.b_1))     # creates the embedded game information.
        self.y0 = tf.add(tf.matmul(self.h, self.W_2), self.b_2)
        self.Y_ = tf.nn.softmax(self.y0)                                # predicted probability distribution of next hero.

        # define loss function:

        # TODO can simplify this part since we know one distribution is one-hot
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.y0, labels=self.Y, name='cross_entropy')
        self.train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.cross_entropy)

    @staticmethod
    def get(args):
        return WinGraph() if args.w else NextHeroGraph()

class NextHeroGraph(DraftGraph):

    def __init__(self):
        super(NextHeroGraph, self).__init__(logitsX=4*N+1, logitsY=N)

    def format(self, game):
        picks_bans = game['picks_bans']
        winning_team = 0 if game['radiant_win'] else 1
        first_pick = picks_bans[0]['team'] # gives a 0 or a 1
        output = []

        team0, team1 = Team(), Team() # team 0 is the winning/picking team

        for i in range(20):
            hero_id = getShiftedID(picks_bans[i]['hero_id'])
            is_pick_bit = 1 if picks_bans[i]['is_pick'] else 0

            if picks_bans[i]['team'] == winning_team:
                a = team0.getContextVector() + team1.getContextVector() + [is_pick_bit]
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

class WinGraph(DraftGraph):

    def __init__(self):
        super(WinGraph, self).__init__(logitsX=2*N, logitsY=1)

    def format(self, game):
        picks_bans = game['picks_bans']
        winning_team = 0 if game['radiant_win'] else 1
        team0, team1 = Team(), Team()
        output = []
        for action in picks_bans:
            if not action['is_pick']: continue
            hero = APIHero.byID(getShiftedID(action['hero_id']))
            if winning_team == action['team']:
                team0.pick(hero)
                output.append((team0.pickVector + team1.pickVector, [1]))
                output.append((team1.pickVector + team0.pickVector, [0])) # is it useful to double up like this?
            else:
                team1.pick(hero)
                output.append((team1.pickVector + team0.pickVector, [1]))
                output.append((team0.pickVector + team1.pickVector, [0]))

        return output

# need this to avoid issues when importing something using argparser
def parseDraftnetArgs():
    argparser = argparse.ArgumentParser(description="Set train and test files.")
    argparser.add_argument('--train', help='path to train file', default='train/pro-7.00.json')
    argparser.add_argument('--dev', help='path to dev file', default='dev/pro-7.00.json')
    argparser.add_argument('--test', help='path to test file', default='test/pro-7.00.json') # FIXME save arguments
    argparser.add_argument('--save', help='path to save model', default="models/pick/bag-{}-{}.ckpt".format(LEARNING_RATE, M))
    argparser.add_argument('--model', help='path to model file', default=None)
    argparser.add_argument('--epochs', help='number of epochs', type=int, default=100)
    argparser.add_argument('--batchSize', help='number of games per batch', type=int, default=100)
    argparser.add_argument('--layer', help='layer to use while rendering TSNE data', type=int, default=2)
    argparser.add_argument('-w', help='use the win prediction graph instead of the next pick graph', action="store_true", default=False)
    argparser.add_argument('--threshold', help='thresold for deciding pick set membership', type=float, default=0.15)
    return argparser.parse_args()

flatten = lambda l: [item for sublist in l for item in sublist]

# create function whose input is a game and output is a list of pairs (the data in the correct format)
def getOneHot(pick):
    return [1 if i == getShiftedID(pick["hero_id"]) else 0 for i in range(N)]

def getNames(picks):
    return [APIHero.byID(pick).getName() for pick in picks]

def getContext(team0, team1, isPick):
    return team0.getContextVector() + team1.getContextVector() + [1 if isPick else 0]

def getDistribution(context, session, graph):
    return session.run(graph.Y_, feed_dict={graph.X: [context]})[0]

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

def loadSession(name):
    session = tf.Session()
    with session.as_default():
        saver = tf.train.Saver()
        saver.restore(session, "models/pick/" + name + ".ckpt")
        return session

def loadSessions(*names):
    return {name: loadSession(name) for name in names}, names

def testInSession(test, session, graph, PICK_THRESHOLD=PICK_THRESHOLD):
    print("starting testing with PICK_THRESHOLD={}..".format(PICK_THRESHOLD))
    counts = [0.0] * 10
    neighborhood_sizes = [0.0] * 10
    for game in test:
        x, y = zip(*graph.format(game))
        distributions = session.run(graph.Y_, feed_dict={graph.X: x, graph.Y: y})
        for i, distribution in enumerate(distributions):
            notAllowed = getNotAllowed(x[i])
            picks = getSuggestions(distribution, notAllowed, PICK_THRESHOLD=PICK_THRESHOLD)
            neighborhood_sizes[i] += len(picks)
            if np.argmax(y[i]) in picks:
                counts[i] += 1
    print("accuracies:", [c / len(test) for c in counts])
    print("neighborhood sizes:", [n / len(test) for n in neighborhood_sizes])

def loadJSONGames(filename):
    return [game for game in json.load(open(filename, "r")) if game["picks_bans"] != None and len(game["picks_bans"]) == 20]

if __name__ == "__main__":

    args = parseDraftnetArgs()
    graph = DraftGraph.get(args)

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if not args.model:

            print("reading training data..")
            train = loadJSONGames(args.train)
            if args.dev != None:
                print("reading dev data..")
                dev = loadJSONGames(args.dev)
                devX, devY = zip(*flatten([graph.format(game) for game in dev]))
            print("building trials..")
            trials = flatten([graph.format(game) for game in train])
            random.shuffle(trials) # since instances from the same game are together

            print("starting training..")
            devLoss = sys.maxint
            for i in range(args.epochs):

                random.shuffle(trials)
                epochLoss = 0.0

                for j in range(0, len(trials) - args.batchSize, args.batchSize):
                    x, y = zip(*trials[j:j + args.batchSize])
                    epochLoss += session.run([graph.cross_entropy, graph.train_step], feed_dict={graph.X: x, graph.Y: y})[0]

                # increasing batch size increases error -- perhaps we should adjust something in the optimization

                print("epoch", i, "mean cross-entropy:", '{:.5f}'.format(sum(epochLoss) / len(trials)))
                saver.save(session, args.save)

                if args.dev == None: break

                newDevLoss = sum(session.run(graph.cross_entropy, feed_dict={graph.X: devX, graph.Y: devY}))
                if newDevLoss > devLoss: break
                devLoss = newDevLoss

        else:
            saver.restore(session, args.model)

        print("reading testing data..")
        test = loadJSONGames(args.test)
        testInSession(test, session, graph, args.threshold)

else:

    graph = NextHeroGraph()

    # need sessionNames to preserve ordering of options
    sessions, sessionNames = loadSessions("pro-7.00-4941", "pub-7.06-3809")
