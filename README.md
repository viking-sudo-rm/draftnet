# Draftnet
A neural net approach to Dota 2 drafting

## What draftnet does

[Dota 2](http://blog.dota2.com/?l=english) is a very popular video game played competitively as an eSport.
An important part of every Dota 2 game is the draft: the phase during which each team of five players picks what 
characters (heroes) they want to play with.

This may sound simple, but drafting well is deceptively complicated. In Dota, each hero has a unique set of abilities. The unique interaction of a hero’s abilities with those of its 
teammates might allow for an awesome combination play. Similarly, another ability might completely mitigate the 
destructive potential of the enemy team. The art of reactively picking a cohesive lineup that counteracts your 
opponent is called drafting, and it is one of the most important parts of any Dota game.

Draftnet learns how to draft well by analyzing thousands of successful drafts by professional Dota players. These games
are read from the [OpenDota](https://www.opendota.com/) API, and then used to train a neural network that makes drafting decisions.

## How to run

Our Python implementation of Draftnet requires the [tensorflow](https://www.tensorflow.org/) machine learning library.

After you have installed tensorflow, you can run the feed-forward network using: 

~~~~
python train.py
~~~~
