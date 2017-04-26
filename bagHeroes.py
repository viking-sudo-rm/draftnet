import tensorflow as tf
import argparse
import json

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
train = [game for game in json.load(open(args.train, "r")) if len(game["picks_bans"]) == 20]

# create function whose input is a game and output is a list of pairs (the data in the correct format)

def format(game):
    data = []
    our_picks = [0 for _ in range(N)]
    our_bans = [0 for _ in range(N)]
    their_picks = [0 for _ in range(N)]
    their_bans = [0 for _ in range(N)]
    for selection in game['picks_bans']:
        






