import numpy as np
import matplotlib
import matplotlib.pyplot as plt

vegetables = [1000*i for i in range(1,6)]
farmers = [100*i for i in range(1,6)]
harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0],
                    [2.4, 0.0, 4.0, 1.0, 2.7],
                    [1.1, 2.4, 0.8, 4.3, 1.9],
                    [0.6, 0.0, 0.3, 0.0, 3.1],
                    [0.7, 1.7, 0.6, 2.6, 2.2]])


fig, ax = plt.subplots()
im = ax.imshow(harvest)

# We want to show all ticks...
ax.set_xticks(np.arange(len(farmers)))
ax.set_yticks(np.arange(len(vegetables)))
# ... and label them with the respective list entries
ax.set_xticklabels(farmers)
ax.set_yticklabels(vegetables)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), ha="right")

# Loop over data dimensions and create text annotations.
for i in range(len(vegetables)):
    for j in range(len(farmers)):
        text = ax.text(j, i, harvest[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Harvest of local farmers (in tons/year)")
ax.colorbar()
fig.tight_layout()
plt.show()