from __future__ import print_function
import requests, json, time, random

N = 9000
WAIT_TIME = 60


def get(url):
    return requests.get(url).json()

if __name__ == "__main__":

	matches = []

	for i in range(0, N, 100):
	    res = get("https://api.opendota.com/api/explorer?sql=select%20match_id,picks_bans,radiant_win%20from%20matches%20where%20game_mode=2%20and%20picks_bans%20is%20not%20null%20order%20by%20match_id%20desc%20limit%20100%20offset%20" + str(i))

	    if "error" in res.keys():
	        print("waiting", WAIT_TIME, "sec for API..")
	        time.sleep(WAIT_TIME)
	        continue

	    matches += res["rows"]
	    print("got", i + 100, "/", N, "matches")

	print("checking downloaded game data..")
	print("# with drafts:", sum(map(lambda x: 0 if x["picks_bans"] == None else 1, matches)))
	print("# total:", len(matches))

	print("saving match data..")
	filename = "data/matches-" + str(len(matches)) + ".json"
	json.dump(matches, open(filename, "w"))

	print("sampling test and train datasets..")
	random.shuffle(matches)
	test = matches[:N // 10]
	train = matches[N // 10:]

	print("saving train dataset..")
	filename = "data/train-" + str(len(train)) + ".json"
	json.dump(train, open(filename, "w"))

	print("saving test dataset..")
	filename = "data/test-" + str(len(test)) + ".json"
	json.dump(test, open(filename, "w"))
