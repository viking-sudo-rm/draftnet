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
		var self = this
		self.list = []
		$http.get("/api/heroes/")
				.then(function (response) {
				    self.list = response.data
		}, function(data) {
                console.log('Error: ' + data);
        })

		self.pickCounter = 0
		const PICK_BAN_ORDER = 	[{"pick": false, "team": 0},  // where the picker is on team 0 (TODO change)
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
								 {"pick": true, "team": 1},
								 {"pick": true, "team": 0}];

		var Team = function() {
			this.picks = []
			this.bans = []
		}

		self.teams = [new Team(), new Team()]

		self.searchFilter = ""
		self.selectedHero = undefined
		self.picked = []

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
			self.selectedHero = hero
		}

		self.isSelected = function(hero) {
			return hero == self.selectedHero
		}

		self.isValidSelection = function() {
			return self.selectedHero && self.picked.indexOf(self.selectedHero) == -1
		}

		//TODO this logic needs to be implemented; this is placeholder
		//TODO hero IDs are wrong (the ones used in Python are down-shifted); import from Python environment instead of from the web
		self.choose = function() {
			var pickBan = PICK_BAN_ORDER[self.pickCounter++]
			self.picked.push(hero)
			self.teams[pickBan.team][pickBan.pick ? "picks" : "bans"].push(hero.id)
			self.predict()
		}

		self.predict = function() {
			data = {
				"team0": self.teams[0],
				"team1": self.teams[1],
				"isPick": PICK_BAN_ORDER[self.pickCounter]["pick"]
			}
			$http.post("/api/predict/", data)
				.then(function successCallback(response) {
				    self.prediction = response.data
				    console.log(self.prediction)
			})
		}

		self.loadHeroes = function() {
			self.byID = {}
			for (var i = 0; i < self.list.length; i++) {
				self.byID[self.list[i].id] = self.list[i]
			}
		}
	});

	console.log("loaded draftnet controller in app.js")

})();