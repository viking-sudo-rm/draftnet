from __future__ import print_function
import json, requests, sys, time, dota2api

API_KEY = "B2666587DD228A597DF0A3B8FC25FCF6"
api = dota2api.Initialise(API_KEY, raw_mode=True)

START_PATCH = 3184637017

# TODO clean this up
# can make this whole process really fast with multi-threading
# store a queue of gameIDs, and pop from it whenever possible
def loadIDsFromDota2API(startMatchId):

	gameIds = set()

	startOfPatch = api.get_match_details(match_id=startMatchId)
	start_patch = startOfPatch["match_seq_num"]
	last_id = int(sys.argv[1]) if len(sys.argv) > 1 else start_patch
	for _ in range(100000):

		try:
			# TODO replace this with get_match_history
			results = api.get_match_history_by_seq_num(game_mode=2, matches_requested=500, start_at_match_seq_num=last_id).values()
			for game in results[1]:
				if game["game_mode"] == 2:
					gameIds.add(game["match_id"])
				last_id = game["match_seq_num"] - 1
				# if game["game_mode"] != 2:
				# 	print("WARNING")
			print("adding results:")
			print("\t#resp ==", len(results[1]))
			print("\t#games ==", len(gameIds))
		except ValueError:
			print(gameIds)
			print("sleeping 60..")
			time.sleep(60)
			print("done sleeping")
		except requests.exceptions.ConnectionError:
			print("connection terminated")
			break

	with open("gameIds.txt", "a") as fh:
		for gameId in gameIds:
			fh.write(str(gameId) + "\n")

	print("last_id:", last_id)

def loadFullGamesFromOpenDota(gameIDs):

	print("starting download..")
	fullGames = []
	for gameID in gameIDs:

		while True:
			response = requests.get("http://api.opendota.com/api/matches/" + str(gameID)).json()
			if "match_id" in response:
				break
			print("\twaiting 60 seconds..")
			time.sleep(60)
			sys.stdout.write("\r\trestarting download..")
			sys.stdout.flush()

		fullGames.append({"match_id": response["match_id"], "picks_bans": response["picks_bans"], "radiant_win": response["radiant_win"]})

		sys.stdout.write("\r\t" + str(len(fullGames)) + " / " + str(len(gameIDs)) + " matches loaded")
		sys.stdout.flush()

		#TODO cut this line
		if len(fullGames) > 3700:
			json.dump(fullGames, open("data/pub-7.06-" + str(len(fullGames)) + ".json", "w"))

	print()
	return fullGames

if __name__ == "__main__":
	print("loading data/ids-3832.json..")
	gameIDs = json.load(open("data/ids-3832.json", "r"))
	fullGames = loadFullGamesFromOpenDota(gameIDs)
	print("saving data/pub-7.06-" + str(len(fullGames)) + ".json..")
	json.dump(fullGames, open("data/pub-7.06-" + str(len(fullGames)) + ".json", "w"))