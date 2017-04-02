import pickle

N = 113

flatten = lambda l: [item for sublist in l for item in sublist]

def getOneHot(pick):
	return [1 if i == pick["hero_id"] - 1 else 0 for i in range(N)]

print "reading training data.."
train = pickle.load(open("data-100000/train-100000.data", "rb"))

X = flatten(map(getOneHot, train["picks_bans"]))
print X
