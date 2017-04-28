from __future__ import print_function
from bagHeroes import *

if __name__ == "__main__":
	with tf.Session() as session:

		# print("loading", args.model + "..")
		# saver = tf.train.Saver()
		# saver.restore(session, args.model)

		team1 = Team()
		am = Hero.byName("antimage")
		print("AM ID:", am.getID())

		team1.pick(am)
		print("AM picked/banned:", am in team1)

		# language specification for user commands:
		# wepick [hero_name]
		# weban [hero_name]
		# theypick [hero_name]
		# theyban [hero_name]
		# after each command, the current neighborhood of likely picks is printed
		# (sorted from highest to lowest probability)
