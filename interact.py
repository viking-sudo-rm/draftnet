from __future__ import print_function
from bagHeroes import *

if __name__ == "__main__":
	with tf.Session() as session:

		print("loading", args.model + "..")
		saver = tf.train.Saver()
		saver.restore(session, args.model)

		while True: # keep going for a bunch of games

			print("started new game")
			team0, team1 = Team(), Team()

			while not team0.isFull() or not team1.isFull(): # keep going until draft is done 

				notAllowed = getNotAllowed(team0.getNotAllowed() + team1.getNotAllowed())
				x = team0.getContextVector() + team1.getContextVector()
				pick_distribution = session.run(Y_, feed_dict={X: [x + [1]]})[0]
				ban_distribution = session.run(Y_, feed_dict={X: [x + [0]]})[0]
				print("picks:", ", ".join(getNames(getPicks(pick_distribution, notAllowed))))
				print("bans:", ", ".join(getNames(getPicks(ban_distribution, notAllowed))))

				action, arg = raw_input("> ").split(" ")
				if action == "wepick":
					team0.pick(Hero.byName(arg))
				elif action == "weban":
					team0.ban(Hero.byName(arg))
				elif action == "theypick":
					team1.pick(Hero.byName(arg))
				elif action == "theyban":
					team1.pick(Hero.byName(arg))
				else:
					break

		# Language specification for user commands:
		# wepick [hero_name]
		# weban [hero_name]
		# theypick [hero_name]
		# theyban [hero_name]
		# after each command, the current neighborhood of likely picks is printed
		# (sorted from highest to lowest probability)

		# Example usage of classes:
		# team0 = Team()
		# am = Hero.byName("antimage")
		# print("AM ID:", am.getID())
		# team0.pick(am)
		# print("AM picked/banned:", am in team0)