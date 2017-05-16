import dota2api, json
API_KEY = "B2666587DD228A597DF0A3B8FC25FCF6"

api = dota2api.Initialise(API_KEY, raw_mode=True)

results = api.get_match_history(game_mode=2).values()
# print results[2]
for game in results[2]:
	print game
	break