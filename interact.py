from __future__ import print_function
from draftnet import *

if __name__ == "__main__":

	args = parseDraftnetArgs()

    with tf.Session() as session:

        print("loading", args.model + "..")
        saver = tf.train.Saver()
        saver.restore(session, args.model)

        # print("reading testing data..")
        # test = [game for game in json.load(open(args.test, "r")) if len(game["picks_bans"]) == 20]
        # testInSession(test, session)

        while True:  # keep going for a bunch of games

            print("started new game")
            team0, team1 = Team(), Team()

            while not team0.isFull() or not team1.isFull():  # keep going until draft is done

                notAllowed = getNotAllowed(team0.getNotAllowed() + team1.getNotAllowed())
                x = team0.getContextVector() + team1.getContextVector()
                pick_distribution = session.run(Y_, feed_dict={X: [x + [1]]})[0]
                ban_distribution = session.run(Y_, feed_dict={X: [x + [0]]})[0]
                print("picks:", ", ".join(getNames(getSuggestions(pick_distribution, notAllowed))))
                print("bans:", ", ".join(getNames(getSuggestions(ban_distribution, notAllowed))))

                action, arg = input("> ").split(" ")
                if action == "wepick":
                    team0.pick(Hero.byName(arg))
                elif action == "weban":
                    team0.ban(Hero.byName(arg))
                elif action == "theypick":
                    team1.pick(Hero.byName(arg))
                elif action == "theyban":
                    team1.ban(Hero.byName(arg))
                else:
                    break

# Language specification for user commands:
# wepick [hero_name]
# weban [hero_name]
# theypick [hero_name]
# theyban [hero_name]
# after each command, the current neighborhood of likely picks is printed
# (sorted from highest to lowest probability)
