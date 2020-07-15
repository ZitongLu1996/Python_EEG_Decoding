# -*- coding: utf-8

import numpy as np
from neurora.rsa_plot import plot_rdm

rdm = np.random.rand(16, 16)
for i in range(16):
    for j in range(16):
        if i>j:
            rdm[i, j] = rdm[j, i]
    rdm[i, i] = 0

conditions = ["0°", "22.5°", "45°", "67.5°", "90°", "112.5°", "135°", "157.5°", "180°",
              "202.5°", "225°", "247.5°", "270°", "292.5°", "315°", "337.5°"]
plot_rdm(rdm, conditions=conditions)