(function() {

	function range(n) {
		return Array.apply(null, Array(n)).map(function (_, i) {return i;});
	}

	var app = angular.module('draftnet', []);

	// use a different symbol for variable text because Django and Angular conflict
	app.config(function($interpolateProvider) {
	  $interpolateProvider.startSymbol('[[');
	  $interpolateProvider.endSymbol(']]');
	});

	app.controller('DraftController', function($scope, $window, $http){

		const N = 113
		const MAX_SUGGESTIONS = 10
		var self = this
		self.list = []
		$http.get("/api/heroes/")
			.then(function (response) {
			    self.list = response.data
			    self.loadHeroes()
		}, function(data) {
                console.log('Error: ' + data);
        })

		$http.get("/api/models/")
			.then(function (response) {
				self.models = response.data
				self.selectedModel = self.models[0]
				if (self.byID) { // predict after both models and heroes have loaded
					self.predict()
				}
			}, function(data) {
				console.log('Error: ' + data)
			})

		self.pickCounter = 0
		const PICK_BAN_ORDER = 	[{"pick": false, "team": 0},  // where team 0 picks first
        						 {"pick": false, "team": 1},
                  				 {"pick": false, "team": 0},
                  				 {"pick": false, "team": 1},

								 {"pick": true, "team": 0},
								 {"pick": true, "team": 1},
								 {"pick": true, "team": 1},
								 {"pick": true, "team": 0},

								 {"pick": false, "team": 1},
								 {"pick": false, "team": 0},
								 {"pick": false, "team": 1},
								 {"pick": false, "team": 0},

								 {"pick": true, "team": 1},
								 {"pick": true, "team": 0},
								 {"pick": true, "team": 1},
								 {"pick": true, "team": 0},

								 {"pick": false, "team": 1},
								 {"pick": false, "team": 0},
								 {"pick": true, "team": 0},
								 {"pick": true, "team": 1}];

		var Team = function() {

			this.picks = []
			this.bans = []

			this.isFull = function() {
				return this.picks.length == 5
			}

		}

		self.teams = [new Team(), new Team()]

		self.searchFilter = ""
		self.selectedHero = undefined
		self.picked = []
		self.banned = []

		self.inTeam = function(hero, team) {
			inList = (x, l) => (l.indexOf(x) != -1)
			return inList(hero, team["picks"]) || inList(hero, team["bans"])
		}

		self.isFiltered = function(hero) {
			if (self.searchFilter == "") {
				return false
			}
			
			var searchText = self.searchFilter.toLowerCase()
			return !hero.name.toLowerCase().includes(searchText) && !hero.localized_name.toLowerCase().includes(searchText)
		}

		self.selectHero = function(hero) {
			if (self.selectedHero == hero) {
				self.selectedHero = undefined
			} else {
				self.selectedHero = hero
			}
		}

		self.isSelected = function(hero) {
			return hero == self.selectedHero
		}

		self.isValidSelection = function() {
			return self.selectedHero && !self.isPicked(self.selectedHero) && !self.isBanned(self.selectedHero)
		}

		self.isPicked = function(hero) {
			return self.picked.indexOf(hero) != -1
		}

		self.isBanned = function(hero) {
			return self.banned.indexOf(hero) != -1
		}

		self.choose = function() {
			var pickBan = PICK_BAN_ORDER[self.pickCounter++]
			if (pickBan.pick) {
				self.picked.push(self.selectedHero)
			} else {
				self.banned.push(self.selectedHero)
			}			
			self.teams[pickBan.team][pickBan.pick ? "picks" : "bans"].push(self.selectedHero.id)
			self.predict(self.getNextAction().team)
			self.searchFilter = ""
		}

		// pass the team you are predicting for
		self.predict = function(team = 0) {

			if (!self.selectedModel) {
				return
			}

			if (self.draftIsDone()) {
				self.prediction = undefined
				return
			}

			data = {
				"team0": team == 0 ? self.teams[0] : self.teams[1],
				"team1": team == 0 ? self.teams[1] : self.teams[0],
				"isPick": self.getNextAction()["pick"],
				"model": self.selectedModel
			}
			$http.post("/api/predict/", data)
				.then(function successCallback(response) {
					self.prediction = []
					for (var i = 0; i < response.data.suggestions.length && i < MAX_SUGGESTIONS; i++) {
						self.prediction.push(self.getHeroByID(response.data.suggestions[i]))
					}
			})
		}

		self.getHeroByID = function (id) {
			return self.byID[id]
		}

		self.getPick = function(team, index){
			return self.getHeroByID(self.teams[team].picks[index])
		}

		self.getBan = function(team, index){
			return self.getHeroByID(self.teams[team].bans[index])
		}

		self.loadHeroes = function() {
			self.byID = {}
			for (var i = 0; i < self.list.length; i++) {
				self.byID[self.list[i].id] = self.list[i]
			}
			if (self.models) { // predict after both models and heroes have loaded
				self.predict()
			}
		}

		// get the next action (pick/ban + team) represented as an object
		self.getNextAction = function() {
			return PICK_BAN_ORDER[self.pickCounter]
		}

		// text to go on the button (either "Ban" or "Pick")
		self.getNextActionTitle = function() {
			return self.getNextAction().pick ? "Pick" : "Ban"
		}

		self.isChoosing = function() {
			return self.getNextAction().team == 0
		}

		self.getSuggestionText = function() {

			if (self.draftIsDone()) {
				return undefined
			}

			var currentTeam = self.getNextAction().team == 0 ? "Team A" : "Team B"
			var currentAction = self.getNextAction().pick ? "picking" : "banning"
			return currentTeam + " should consider " + currentAction + ":"
		}

		self.getTurnText = function() {
			var currentTeam = self.getNextAction().team == 0 ? "Team A's" : "Team B's"
			var currentAction = self.getNextAction().pick ? "pick" : "ban"
			return currentTeam + " turn to " + currentAction
		}

		self.draftIsDone = function() {
			return self.teams[0].isFull() && self.teams[1].isFull()
		}

	});

	console.log("loaded draftnet controller in app.js")

})();