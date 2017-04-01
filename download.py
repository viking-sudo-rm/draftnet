import dota2api, pickle, time

KEY = "B2666587DD228A597DF0A3B8FC25FCF6"

api = dota2api.Initialise(KEY)

# leagueListing = api.get_league_listing()
# print len(leagueListing["leagues"])
# print leagueListing["leagues"][0]

def downloadMatches(nextMatch, n, matchType):

	allMatches = []
	for i in range (n / 100):
		try:
			matchHistory = api.get_match_history_by_seq_num(start_at_match_seq_num=nextMatch, **matchType)
			nextMatch = matchHistory["matches"][99]["match_seq_num"] + 1
			allMatches += matchHistory["matches"]
		except ValueError:
			print "[WAIT] max API requests exceeded"
			time.sleep(60)
			break

	return allMatches, nextMatch

# start of 7.0.0 -- match_id: 2841970697

GAMES = 10000
FILENAME = "matches-" + str(GAMES) + ".data"
TYPE = {
	"game_mode": 2, #captain's mode
	"lobby_type": 2, #tournament
}

if __name__ == "__main__":

	t = api.get_match_details(2841970697)["match_seq_num"]
	startT = t
	allMatches = []
	while len(allMatches) < GAMES:
		print "got", len(allMatches), "/", GAMES, "matches"
		matches, t = downloadMatches(t,100, TYPE)
		allMatches += matches

	print "last match:", t
	print "saving results to " + FILENAME + ".."
	pickle.dump(allMatches, open(FILENAME, "wb"))