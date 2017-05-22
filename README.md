# Draftnet

[Dota 2](http://blog.dota2.com/?l=english) is a popular video game played competitively as an eSport.
An important part of every Dota 2 game is the draft: the phase during which each team of five players picks what 
heroes (characters) they want to play with.

This may sound simple, but drafting well is deceptively complicated. Each Dota hero has a unique set of abilities, and the  specific interaction of one hero’s abilities with those of its 
teammates might allow for an awesome combination play. Similarly, another ability might completely mitigate the 
destructive potential of the enemy team. Among professional Dota players, drafting is complicated artform that decides the outcomes of million-dollar games.

Draftnet learns how to draft well by analyzing thousands of winning drafts by professional Dota players. These games
are read from the [OpenDota](https://www.opendota.com/) API, and then used to train a feed-forward neural network that makes drafting decisions.

## Using the Draftnet web interface
You can find an easy-to-use web interface for Draftnet at http://draftnet.herokuapp.com/. The website is designed so that you can have it open in a secondary monitor or tab while you are picking a Dota 2 draft. Because of temporary limitations on the OpenDota API, the network currently only predicts picks and bans for Captain's Mode games.

## Running the Draftnet source code

Our Python implementation of Draftnet requires the [Tensorflow](https://www.tensorflow.org/) machine learning library. After you have installed Tensorflow and cloned our repository, you can train our feed-forward network with `draftnet.py`. For example:

~~~~
python draftnet.py --train train/pro-7.00.json --test test/pro-7.00.json --save results/customModel.ckpt
~~~~

Train and test files contain the results of Dota 2 games fetched from the OpenDota API. Type `python draftnet.py --help` to view a full list of run options.

## How Draftnet works

Draftnet uses a "bag-of-heroes" architecture inspired by feed-forward neural networks like Word2Vec that learn vector semantics for words. The network takes as input four sets of heroes (the picks and bans on each team) and one bit that encodes whether the next draft action is a pick or ban. Via logistic regression, these inputs feed into a hidden layer with ~50 bits, which in turn feeds into a final output layer with 113 bits (corresponding to the number of heroes in Dota). The softmax of this output layer is interpretted as a probability distribution for the next pick or ban. During training, the weights and biases are then updated using gradient descent. The loss function in this optimization is the cross entropy between the network's predicted distribution and the one-hot distribution encoding the actual pick or ban.

## Hero embeddings

In addition to predicting hero picks, our neural network also produces embeddings of each hero in Dota. Intuitively, an embedding is a vector representation of a hero that captures all the important features that that hero has. Since the network stores these embeddings as vectors, we can use them to visualize the relationships between heroes. You can use `tsne.py` to produce a plot of these hero vectors in 2D space. For example:

~~~
python tsne.py --model results/bag-100-1000000-0.01-50.ckpt
~~~
