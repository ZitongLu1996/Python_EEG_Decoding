# -*- coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

rdm = np.random.rand(16, 16)
for i in range(16):
    rdm[i, i] = 0

plt.imshow(rdm, extent=(0, 1, 0, 1), cmap=plt.cm.jet, clim=(0, 1))

# plt.axis("off")
cb = plt.colorbar()
cb.ax.tick_params(labelsize=16)
font = {'size': 18}

cb.set_label("Dissimilarity", fontdict=font)
step = float(1 / 16)
x = np.arange(0.5 * step, 1 + 0.5 * step, step)
y = np.arange(1 - 0.5 * step, -0.5 * step, -step)
conditions = ["0°", "22.5°", "45°", "67.5°", "90°", "112.5°", "135°", "157.5°", "180°",
              "202.5°", "225°", "247.5°", "270°", "292.5°", "315°", "337.5°"]
plt.xticks(x, conditions, fontsize=12, rotation=30, ha="right")
plt.yticks(y, conditions, fontsize=12)
plt.xlabel("Time 1", fontsize=16)
plt.ylabel("Time 2", fontsize=16)
plt.show()