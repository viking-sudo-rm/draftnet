import tensorflow as tf
import argparse
import json
from display import *
from operator import add
N = 113     # number of heroes
M = 25
LEARNING_RATE = 0.001
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
y_ = tf.sigmoid(y0)                                # probability distribution of next hero.

# define loss function:

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y0, labels=Y, name='cross_entropy')
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# load games

argparser = argparse.ArgumentParser(description="Set train and test files.")
argparser.add_argument('--train', help='path to train file', default='data/train-8100.json')
argparser.add_argument('--test', help='path to test file', default='data/test-900.json')
args = argparser.parse_args()
train = [game for game in json.load(open(args.test, "r")) if len(game["picks_bans"]) == 20]



# create function whose input is a game and output is a list of pairs (the data in the correct format)


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
        if picks_bans[i]['team'] == 0:
            a = team0picks + team0bans + team1picks + team1bans
            output.append((a, getOneHot(picks_bans[i])))
            if picks_bans[i]['is_pick']:
                team0picks = list(map(add, team0picks, getOneHot(picks_bans[i])))
            else:
                team0bans = list(map(add, team0bans, getOneHot(picks_bans[i])))
        else:
            a = team1picks + team1bans + team0picks + team0bans
            output.append((a, getOneHot(picks_bans[i])))
            if picks_bans[i]['is_pick']:
                team1picks = list(map(add, team1picks, getOneHot(picks_bans[i])))
            else:
                team1bans = list(map(add, team1bans, getOneHot(picks_bans[i])))
    return output

