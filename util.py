from __future__ import print_function
import requests

N = 113     # number of heroes

PICK_BAN_ORDER = [	(False, 0),  # where the picker is on team 0
					(False, 1),
					(False, 0),
					(False, 1),
					(True, 0),
					(True, 1),
					(True, 1),
					(True, 0),
					(False, 1),
					(False, 0),
					(False, 1),
					(False, 0),
					(True, 1),
					(True, 0),
					(True, 1),
					(True, 0),
					(False, 1),
					(False, 0),
					(True, 1)	] # something is fucked up with the order

class Hero:

	def __init__(self, json):
		self.json = json

	def getID(self):
		return self.json["id"]

	def getName(self):
		return self.json["localized_name"]

	@classmethod
	def byID(cls, id):
		return cls.heroes[id]

	@classmethod
	def byName(cls, name):
		return cls.heroByName[cls.getPlainName(name)]

	@staticmethod
	def getPlainName(name):
		# TODO use regex here
		return name.lower().replace(" ", "").replace("_", "").replace("-", "")

# TODO store heroes by ID
print("downloading hero data..")
Hero.heroes = requests.get("https://api.opendota.com/api/heroes").json()

# maps from apiID -> localID
_idMap = {Hero.heroes[i]["id"] : i for i in range(len(Hero.heroes))}
_idMapInv = {i : Hero.heroes[i]["id"] for i in range(len(Hero.heroes))}

def getShiftedID(apiID):
	return _idMap[apiID]

def getUnshiftedID(localID):
	return _idMapInv[localID]

for hero in Hero.heroes:
	hero["id"] = getShiftedID(hero["id"])

Hero.heroes = [Hero(hero) for hero in Hero.heroes]
Hero.heroByName = {Hero.getPlainName(hero.getName()) : hero for hero in Hero.heroes}

def getVectorForSet(heroSet):
	return [1 if hero in heroSet else 0 for hero in Hero.heroes]

class Team:

	MAX_PICKS = 5

	# should add references to Hero objects to these sets
	def __init__(self):

		# sets holding hero objects
		self.picks = set()
		self.bans = set()
		# TODO change these sets to ordered lists

		# vectors with v_i = 1 iff hero i is picked/banned
		self.pickVector = [0] * N
		self.banVector = [0] * N

	def pick(self, hero):
		if self.isFull(): return False
		self.picks.add(hero)
		self.pickVector[hero.getID()] = 1
		return True

	def ban(self, hero):
		self.bans.add(hero)
		self.banVector[hero.getID()] = 1

	def isFull(self):
		return len(self.picks) == Team.MAX_PICKS

	def getNotAllowed(self):
		union = self.picks | self.bans
		return [True if h in union else False for h in Hero.heroes]

	def getContextVector(self):
		return self.pickVector + self.banVector

	def __contains__(self, hero):
		return hero in self.picks or hero in self.bans

	@staticmethod
	def getContextVectorFor(team0, team1, pickBit):
		team0.getContextVector() + team1.getContextVector() + [pickBit]

	# TODO define methods to get neural-net inputs from a Team instance