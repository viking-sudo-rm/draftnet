import dota2api, pickle, time

KEY = "B2666587DD228A597DF0A3B8FC25FCF6"

api = dota2api.Initialise(KEY)

# leagueListing = api.get_league_listing()
# print len(leagueListing["leagues"])
# print leagueListing["leagues"][0]

def downloadMatches(nextMatch, n, filename):

	allMatches = []
	for i in range (n / 100):
		try:
			matchHistory = api.get_match_history_by_seq_num(start_at_match_seq_num=nextMatch)
			nextMatch = matchHistory["matches"][99]["match_seq_num"] + 1
			allMatches += matchHistory["matches"]
		except ValueError:
			print "[WAIT] max API requests exceeded"
			break

	print "next match: " + str(nextMatch)
	return allMatches, nextMatch

# start of 7.0.0 -- match_id: 2841970697

FILENAME = "matches.data"

if __name__ == "__main__":

	t = api.get_match_details(2841970697)["match_seq_num"]
	startT = t
	allMatches = []
	for i in range(1000):
		print "got", t - startT, "matches"
		matches, t = downloadMatches(t,100,"matches-" + str(t) + ".data")
		allMatches += matches

	print "saving results to " + FILENAME + ".."
	pickle.dump(allMatches, open(FILENAME, "wb"))