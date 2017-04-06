from __future__ import print_function
import requests

PICK_BAN_ORDER = [	(False, 1),  # where the picker is on team 0
					(False, 0),
					(False, 1),
					(False, 0),
					(True, 1),
					(True, 0),
					(True, 0),
					(True, 1),
					(False, 0),
					(False, 1),
					(False, 0),
					(False, 1),
					(True, 0),
					(True, 1),
					(True, 0),
					(True, 1),
					(False, 0),
					(False, 1),
					(True, 1)	]

print("downloading hero data..")
heroes = requests.get("https://api.opendota.com/api/heroes").json()

# maps from apiID -> localID
idMap = {heroes[i]["id"] : i for i in range(len(heroes))}
idMapInv = {i : heroes[i]["id"] for i in range(len(heroes))}

def getShiftedID(apiID):
	return idMap[apiID]

def getUnshiftedID(localID):
	return idMapInv[localID]

for hero in heroes:
	hero["id"] = getShiftedID(hero["id"])

roles =[]
for i in range(len(heroes)):
	roles += heroes[i]['roles']

roles = list(set(roles))


def vec2hero(vector):
	for i in range(len(heroes)):
		if vector[i] == 1:
			return int2hero(i)
	return None


def int2hero(i):
	return heroes[i]


def getName(hero):
	if hero == None:
		return "Error"
	return hero["localized_name"]


def heroes2str(l):
	return ", ".join([getName(h) for h in l])


class Team:

	def __init__(self, name):
		self.name = name
		self.picks = []
		self.bans = []

	def __str__(self):
		return self.name + ":\n\tpicks: " + heroes2str(self.picks) + "\n\tbans: " + heroes2str(self.bans)

	def pick(self, hero):
		self.picks.append(hero)

	def ban(self, hero):
		self.bans.append(hero)

	def isValid(self, hero):
		return hero not in self.bans + self.picks