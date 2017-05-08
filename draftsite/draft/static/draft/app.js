(function() {

	function range(n) {
		return Array.apply(null, Array(n)).map(function (_, i) {return i;});
	}

	var app = angular.module('draftnet', []);

	app.controller('DraftController', function($scope, $window, $http){

		const N = 113
		var self = this

		makeTeam = function() {
			return {"picks": [], "bans": []}
		}

		this.team0 = makeTeam()
		this.team1 = makeTeam()

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

		this.isTaken = function(heroId) {
			return this.inTeam(heroId, this.team0) || this.inTeam(heroId, this.team1)
		}

		this.selectHero = function(hero) {
			self.selectedHero = hero
		}

		//TODO this logic needs to be implemented; this is placeholder
		//TODO hero IDs are wrong (the ones used in Python are down-shifted); import from Python environment instead of from the web
		this.pick = function() {
			var id = document.getElementById(self.selectedHero).getAttribute("hero-id")
			self.team0["picks"].push(id)
			console.log(id)
		}

		this.ban = function() {
			var id = document.getElementById(self.selectedHero).getAttribute("hero-id")
			self.team0["bans"].push(id)
		}

	});

	console.log("loaded draftnet controller in app.js")

})();