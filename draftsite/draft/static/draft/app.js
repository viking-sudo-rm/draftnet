(function() {

	function range(n) {
		return Array.apply(null, Array(n)).map(function (_, i) {return i;});
	}

	var app = angular.module('draftnet', []);

	// use a different symbol for variable text because Django and Angular conflict
	app.config(function($interpolateProvider) {
	  $interpolateProvider.startSymbol('{[{');
	  $interpolateProvider.endSymbol('}]}');
	});

	app.controller('DraftController', function($scope, $window, $http){

		const N = 113
		var self = this

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

		makeTeam = function() {
			return {"picks": [], "bans": []}
		}

		this.teams = [makeTeam(), makeTeam()]

		this.searchFilter = ""
		this.selectedHero = undefined

		this.inTeam = function(id, team) {
			inList = (x, l) => (l.indexOf(x) != -1)
			return inList(id, team["picks"]) || inList(id, team["bans"])
		}

		this.isFiltered = function(hero) {
			if (self.searchFilter == "") {
				return false
			}
			var heroName = document.getElementById(hero).name
			var searchText = this.searchFilter.toLowerCase()
			return !hero.toLowerCase().includes(searchText) && !heroName.toLowerCase().includes(searchText)
		}

		this.selectHero = function(hero) {
			self.selectedHero = hero

		}

		this.isSelected = function(hero) {
			return hero == self.selectedHero
		}

		this.getSelectedElement = function() {
			return angular.element(document.querySelector("#" + self.selectedHero));
		}

		this.isValidSelection = function() {
			var id = self.getSelectedElement().hasClass()
			return self.selectedHero && !self.getSelectedElement().hasClass("taken")
		}

		//TODO this logic needs to be implemented; this is placeholder
		//TODO hero IDs are wrong (the ones used in Python are down-shifted); import from Python environment instead of from the web
		this.choose = function() {
			var pickBan = PICK_BAN_ORDER[self.pickCounter++]
			var element = self.getSelectedElement();
			element.addClass("taken");
			self.teams[pickBan.team][pickBan.pick ? "picks" : "bans"].push(parseInt(element.attr("hero-id")))
			self.predict()
		}

		this.predict = function() {
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

		this.loadHeroes = function() {
			self.byID = {}
			heroes = $('#heroList .grid').map(function(){

				self.byID[$(this).attr('hero-id')] = this
			}).get();
			console.log(self.byID)
		}

		this.getImageForID = function(id) {
			return self.byID[id].getAttribute("src")
		}

	});

	console.log("loaded draftnet controller in app.js")

})();