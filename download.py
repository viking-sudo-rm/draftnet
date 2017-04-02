import requests, pickle, time, random

N = 100000
FILENAME = "matches-" + str(N) + ".data"
TEST_NAME = "test-" + str(N) + ".data"
TRAIN_NAME = "train-" + str(N) + ".data"
WAIT_TIME = 60

def get(url):
	return requests.get(url).json()

matches = []

i = 0
while i < N:
	res = get("https://api.opendota.com/api/explorer?sql=select%20match_id,picks_bans,radiant_win%20from%20matches%20where%20game_mode=2%20and%20picks_bans%20is%20not%20null%20order%20by%20match_id%20desc%20limit%20100%20offset%20" + str(i))
	
	if "error" in res.keys():
		print "waiting", WAIT_TIME, "sec for API.."
		time.sleep(WAIT_TIME)
		continue

	matches += res["rows"]
	i += 100
	print "got", i, "/", N, "matches"

print "checking downloaded game data.."
print "# with drafts:", sum(map(lambda x: 0 if x["picks_bans"] == None else 1, matches))
print "# total:", len(matches)

print "saving to", FILENAME + ".."
pickle.dump(matches, open(FILENAME, "wb"))

print "sampling test and train datasets.."
random.shuffle(matches)
test = matches[:N / 10]
train = matches[N / 10:]

print "saving test dataset.."
pickle.dump(test, open(TEST_NAME, "wb"))
print "saving train dataset.."
pickle.dump(train, open(TRAIN_NAME, "wb"))