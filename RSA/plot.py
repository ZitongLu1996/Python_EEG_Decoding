import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from neurora.stuff import permutation_test

f = h5py.File("corrs/ERP.h5", "r")
corrs = np.array(f["ori"])
rs = corrs[:, :, 0]

for sub in range(16):
    for t in range(100):
        if t<=1:
            rs[sub, t] = np.average(rs[sub, :t+3])
        if t>1 and t<98:
            rs[sub, t] = np.average(rs[sub, t-2:t+3])
        if t>=98:
            rs[sub, t] = np.average(rs[sub, t-2:])

avg = np.average(rs, axis=0)
err = np.zeros([100], dtype=np.float)
for t in range(100):
    err[t] = np.std(rs[:, t], ddof=1)/np.sqrt(16)

ps = np.zeros([100], dtype=np.float)
chance = np.full([16], 0)
for t in range(100-35):
    ps[t+35] = permutation_test(rs[:, t+35], chance)
    if ps[t+35] < 0.05 and avg[t+35] > 0:
        plt.plot(t * 0.02 + 0.2, 0.42, "s", color="orangered", alpha=1)
        xi = [t * 0.02 + 0.2, t * 0.02 + 0.02 + 0.2]
        ymin = [0]
        ymax = [avg[t+35]-err[t+35]]
        plt.fill_between(xi, ymax, ymin, facecolor="orangered", alpha=0.1)

ax=plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
ax.spines['bottom'].set_position(('data', 0))

x = np.arange(-0.5+0.012, 1.5+0.012, 0.02)
plt.fill_between(x, avg+err, avg-err, facecolor="orangered", alpha=0.8)
plt.fill_between([0, 0.2], -1, 1, facecolor="grey", alpha=0.1)
plt.ylim(-0.04*3, 0.15*3)
plt.xlim(-0.5, 1.5)
plt.xticks([-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
plt.tick_params(labelsize=12)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Representational Similarity", fontsize=16)
plt.show()