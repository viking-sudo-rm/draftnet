(function() {

	function range(n) {
		return Array.apply(null, Array(n)).map(function (_, i) {return i;});
	}

	var app = angular.module('draftnet', []);

	app.controller('DraftController', function($scope, $window, $http){

		const N = 113
		var self = this

		var pickCounter = 0
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

		this.loadHeroes = function() {
			self.heroes = $('#heroList .grid').map(function(){
				return $(this).attr('id');
			}).get();
			console.log(self.heroes)
		}

		this.inTeam = function(heroId, team) {
			inList = (x, l) => (l.indexOf(x) != -1)
			return inList(heroId, team["picks"]) || inList(heroId, team["bans"])
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

		//TODO this logic needs to be implemented; this is placeholder
		//TODO hero IDs are wrong (the ones used in Python are down-shifted); import from Python environment instead of from the web
		this.choose = function() {
			var pickBan = PICK_BAN_ORDER[pickCounter++]
			var element = angular.element(document.querySelector("#" + self.selectedHero));
			element.addClass("taken");
			console.log(element)
			console.log(element.attr("hero-id"))
			self.teams[pickBan.team][pickBan.pick ? "picks" : "bans"].push(element.attr("hero-id"))
			console.log(self.teams)
		}

	});

	console.log("loaded draftnet controller in app.js")

})();