(function() {

	function range(n) {
		return Array.apply(null, Array(n)).map(function (_, i) {return i;});
	}

	var app = angular.module('draftnet', []);

	app.controller('DraftController', function($scope, $window, $http){

		const N = 113
		var self = this

		this.selectedHeroes = []
		this.searchFilter = ""

		this.updateHeroSelections = function() {
			console.log("update heroes here")
		}

	});

	console.log("loaded draftnet controller in app.js")

})();