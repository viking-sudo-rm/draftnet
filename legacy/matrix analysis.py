import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import display


picks = np.loadtxt("results/W_p.csv", delimiter=",")
bans = np.loadtxt("results/W_b.csv", delimiter=",")

print(np.linalg.norm(picks))
print(np.linalg.norm(bans))
print(np.linalg.norm(picks-bans))