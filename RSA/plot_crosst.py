import matplotlib.pyplot as plt
import h5py
import numpy as np

subs = ["201", "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "212", "213", "215", "216", "217", "218"]

rlts = np.zeros([16, 100, 100], dtype=np.float)

subindex = 0
for sub in subs:
    f = h5py.File("corrs_crosst/ERP_"+sub+".h5", "r")
    rlts[subindex] = np.array(f["ori"])[:, :, 0]
    f.close()
    subindex = subindex + 1

avg = np.average(rlts, axis=0)

ax=plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
plt.fill_between([0, 0.2], -0.5, 1.5, facecolor="grey", alpha=0.2)
plt.fill_between([-0.5, 0], 0, 0.2, facecolor="grey", alpha=0.2)
plt.fill_between([0.2, 1.5], 0, 0.2, facecolor="grey", alpha=0.2)
plt.imshow(avg, extent=(-0.5, 1.5, -0.5, 1.5), origin='low', cmap="bwr", clim=(-0.09, 0.09))
cb=plt.colorbar(ticks=[-0.07, 0.07])
cb.ax.tick_params(labelsize=12)
font = {'size': 15,}
cb.set_label('Representational Similarity', fontdict=font)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Training Time-point (s)", fontsize=16)
plt.ylabel("Test Time-point (s)", fontsize=16)
plt.show()