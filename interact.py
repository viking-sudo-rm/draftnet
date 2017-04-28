from __future__ import print_function
from bagHeroes import *

if __name__ == "__main__":
	with tf.Session() as session:

		print("loading", args.model + "..")
		saver = tf.train.Saver()
		saver.restore(session, args.model)

		while True: # keep going for a bunch of games

			print("started new game")
			team1, team2 = Team(), Team()

			while not team1.isFull() or not team2.isFull(): # keep going until draft is done 

				action, arg = raw_input("> ").split(" ")
				if action == "wepick":
					team1.pick(Hero.byName(arg))
				elif action == "weban":
					team1.ban(Hero.byName(arg))
				elif action == "theypick":
					team2.pick(Hero.byName(arg))
				elif action == "theyban":
					team2.pick(Hero.byName(arg))
				else:
					break

				notAllowed = getNotAllowed(team1.getNotAllowed() + team2.getNotAllowed())
				x = team1.getContextVector() + team2.getContextVector()
				pick_distribution = session.run(Y_, feed_dict={X: [x + [1]]})[0]
				ban_distribution = session.run(Y_, feed_dict={X: [x + [0]]})[0]
				print("picks:", ", ".join(getNames(getPicks(pick_distribution, notAllowed))))
				print("bans:", ", ".join(getNames(getPicks(ban_distribution, notAllowed))))

		# Example usage of classes:
		# team1 = Team()
		# am = Hero.byName("antimage")
		# print("AM ID:", am.getID())
		# team1.pick(am)
		# print("AM picked/banned:", am in team1)

		# Language specification for user commands:
		# wepick [hero_name]
		# weban [hero_name]
		# theypick [hero_name]
		# theyban [hero_name]
		# after each command, the current neighborhood of likely picks is printed
		# (sorted from highest to lowest probability)

		# TODO should add null prediction with 0 for all heroes