import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.stats import ttest_1samp
from neurora.stuff import permutation_test

subs = ["201", "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "212", "213", "215", "216", "217", "218"]

rlts = np.zeros([16, 100, 100], dtype=np.float)

subindex = 0
f = h5py.File("svm_results/ERP_ori.h5", "r")
for sub in subs:
    rlts[subindex] = np.array(f[sub])
    subindex = subindex + 1
f.close()

avg = np.average(rlts, axis=0)

"""ps = np.zeros([65, 65], dtype=np.float)
chance = np.full([16], 0.0625)
for i in range(65):
    for j in range(65):
        p = permutation_test(rlts[:, i+35, j+35], chance)
        if p < 0.05 and avg[i+35, j+35] > 0.0625:
            ps[i, j] = 1
        else:
            ps[i, j] = 0

newps = np.zeros([102-35, 102-35])
newps[1:101-35, 1:101-35] = ps"""

ax=plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
plt.fill_between([0, 0.2], -0.5, 1.5, facecolor="grey", alpha=0.2)
plt.fill_between([-0.5, 0], 0, 0.2, facecolor="grey", alpha=0.2)
plt.fill_between([0.2, 1.5], 0, 0.2, facecolor="grey", alpha=0.2)
plt.imshow(avg, extent=(-0.5, 1.5, -0.5, 1.5), origin='low', cmap="bwr", clim=(0.045, 0.08))
cb=plt.colorbar(ticks=[0.05, 0.075])
cb.ax.tick_params(labelsize=12)
#x = np.linspace(-0.5-0.01+0.7, 1.5+0.01, 102-35)
#y = np.linspace(-0.5-0.01+0.7, 1.5+0.01, 102-35)
#X, Y = np.meshgrid(x, y)
#plt.contour(X, Y, newps, (0.5), colors='dimgrey', linewidths=1.5)
font = {'size': 15,}
cb.set_label('Classification Accuracy', fontdict=font)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Training Time-point (s)", fontsize=16)
plt.ylabel("Test Time-point (s)", fontsize=16)
plt.show()