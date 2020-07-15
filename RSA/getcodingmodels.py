import numpy as np

rdm = np.zeros([16, 16], dtype=np.float)

for i in range(16):
    for j in range(16):
        diff = np.abs(i-j)
        if diff <= 8:
            rdm[i, j] = diff/8
        else:
            rdm[i, j] = (16-diff)/8

from neurora.rsa_plot import plot_rdm

conditions = ["0°", "22.5°", "45°", "67.5°", "90°", "112.5°", "135°", "157.5°", "180°",
              "202.5°", "225°", "247.5°", "270°", "292.5°", "315°", "337.5°"]

plot_rdm(rdm, conditions=conditions)

np.savetxt("modelrdm/modelrdm.txt", rdm)
