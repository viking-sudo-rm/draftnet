# Draftnet
A neural net approach to Dota 2 drafting

## What draftnet does

[Dota 2](http://blog.dota2.com/?l=english) is a very popular video game played competitively as an eSport.
An important part of every Dota 2 game is the draft: the phase during which each team of five players picks what 
characters (heroes) they want to play with.

This may sound simple, but drafting well is deceptively complicated. In Dota, each hero has a unique set of abilities. The unique interaction of a hero’s abilities with those of its 
teammates might allow for an awesome combination play. Similarly, another ability might completely mitigate the 
destructive potential of the enemy team. Among professional Dota players, drafting is complicated artform that decides the fate of million-dollar games.

Draftnet learns how to draft well by analyzing thousands of successful drafts by professional Dota players. These games
are read from the [OpenDota](https://www.opendota.com/) API, and then used to train a neural network that makes drafting decisions.

## How to run

Our Python implementation of draftnet requires the [tensorflow](https://www.tensorflow.org/) machine learning library. After you have installed tensorflow, you can run our feed-forward network by running:

~~~~
python train.py --train data/train-17900.json --test data/test-2000.json
~~~~

## Hero embeddings

In addition to predicting hero picks, our neural network also produces embeddings of each hero in DotA. Intuitively, an embedding is a vector representation of a hero that captures all the important features that that hero has. Since the network stores these embeddings as vectors, we can use them to visualize the relationships between heroes. Every time you run `train.py`, the resulting weights will be exported to a file in the results folder. You can use `tsne.py` to produce a plot of these hero vectors in 2D space.
