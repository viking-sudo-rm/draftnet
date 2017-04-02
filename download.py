import urllib3

http = urllib3.PoolManager()

def get(url):
	r = http.request("GET", url)
	return r.data

print get("http://api.opendota.com/api/proMatches")
# print get("https://api.opendota.com/api/explorer?sql=select%20picks_bans,radiant_win%20from%20matches%20order%20by%20match_id%20desc%20limit%20100%20offset%20100")