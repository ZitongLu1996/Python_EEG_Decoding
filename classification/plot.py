import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from neurora.stuff import permutation_test

subs = ["201", "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "212", "213", "215", "216", "217", "218"]
nsubs = len(subs)

f = h5py.File("svm_results/ERP_ori.h5", "r")
rlts = np.zeros([nsubs, 100], dtype=np.float)

subindex = 0
for sub in subs:
    subrlts = np.array(f[sub])
    for t in range(100):
        rlts[subindex, t] = subrlts[t, t]

    for t in range(100):
        if t<=1:
            rlts[subindex, t] = np.average(rlts[subindex, :t+3])
        if t>1 and t<98:
            rlts[subindex, t] = np.average(rlts[subindex, t-2:t+3])
        if t>=98:
            rlts[subindex, t] = np.average(rlts[subindex, t-2:])

    subindex = subindex + 1

f.close()

avg = np.average(rlts, axis=0)
err = np.zeros([100], dtype=np.float)
for t in range(100):
    err[t] = np.std(rlts[:, t], ddof=1)/np.sqrt(nsubs)
print(avg.shape)
print(err.shape)

ps = np.zeros([100], dtype=np.float)
chance = np.full([16], 0.0625)
for t in range(100-35):
    ps[t+35] = permutation_test(rlts[:, t+35], chance)
    if ps[t+35] < 0.05 and avg[t+35] > 0.0625:
        plt.plot(t * 0.02 + 0.2, 0.148, "s", color="orangered", alpha=0.8)
        xi = [t * 0.02 + 0.2, t * 0.02 + 0.02 + 0.2]
        ymin = [0.0625]
        ymax = [avg[t+35]-err[t+35]]
        plt.fill_between(xi, ymax, ymin, facecolor="orangered", alpha=0.15)

ax=plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(3)
ax.spines["bottom"].set_linewidth(3)
ax.spines['bottom'].set_position(('data', 0.0625))

x = np.arange(-0.5+0.008, 1.5+0.008, 0.02)

#plt.plot(x, avg)
plt.fill_between(x, avg+err, avg-err, facecolor="orangered", alpha=0.8)
plt.fill_between([0, 0.2], -1, 1, facecolor="grey", alpha=0.1)
#plt.axhline(y=0.0625, c="darkgrey", alpha=0.7, ls="--", linewidth="5")

plt.ylim(0.05, 0.15)
plt.xlim(-0.5, 1.5)
plt.xticks([-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
plt.tick_params(labelsize=12)
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Classification Accuracy", fontsize=16)
plt.show()

