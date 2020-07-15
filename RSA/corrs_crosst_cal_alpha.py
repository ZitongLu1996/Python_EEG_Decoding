import numpy as np
from neurora.rdm_cal import eegRDM
from neurora.stuff import limtozero
from scipy.stats import pearsonr, spearmanr
import h5py
import time

def crosstime_corrs_cal(eegdata, modelrdm, time_win=5, time_step=5):
    # shape of data: [n_conditions, n_trials, n_channels, n_times]
    cons, trials, chls, ts = np.shape(eegdata)
    ts = int((ts - time_win) / time_step) + 1

    data = np.zeros([ts, cons, chls, time_win], dtype=np.float)
    for j in range(ts):
        for k in range(cons):
            for l in range(chls):
                for m in range(time_win):
                    data[j, k, l, m] = np.average(eegdata[k, :, l, j * time_step + m])
    data = np.reshape(data, [ts, cons, chls * time_win])

    corrs = np.zeros([ts, ts, 2], dtype=np.float)

    v2 = np.zeros([cons*(cons-1)], dtype=np.float)
    index = 0
    for i in range(cons):
        for j in range(cons):
            if i != j:
                v2[index] = modelrdm[i, j]
                index = index + 1

    for t1 in range(ts):
        print(t1)
        for t2 in range(ts):
            v1 = np.zeros([cons*(cons-1)], dtype=np.float)
            index = 0
            for k in range(cons):
                for l in range(cons):
                    if k != l:
                        r = pearsonr(data[t1, k], data[t2, l])[0]
                        v1[index] = limtozero(1 - abs(r))
                        index = index + 1
            corrs[t1, t2] = spearmanr(v1, v2)

    return corrs


subs = ["201", "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "212", "213", "215", "216", "217", "218"]

nsubs = len(subs)

modelrdm = np.loadtxt("modelrdm/modelrdm.txt")

"""starttime = time.clock()

for sub in subs:
    f = h5py.File("data_for_RSA/ERP/" + sub + ".h5", "r")
    ori_subdata = np.array(f["ori"])
    pos_subdata = np.array(f["pos"])
    f.close()
    f = h5py.File("corrs_crosst/ERP_"+str(sub)+".h5", "w")
    ori_eegrdms = crosstime_corrs_cal(ori_subdata, modelrdm)
    f.create_dataset("ori", data=ori_eegrdms)
    pos_eegrdms = crosstime_corrs_cal(pos_subdata, modelrdm)
    f.create_dataset("pos", data=pos_eegrdms)
    f.close()

runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("corrs_crosst/ERP_runtime.txt", runtime)"""


starttime = time.clock()

for sub in subs:
    f = h5py.File("data_for_RSA/Alpha/" + sub + ".h5", "r")
    ori_subdata = np.array(f["ori"])
    pos_subdata = np.array(f["pos"])
    f.close()
    f = h5py.File("corrs_crosst/Alpha_"+str(sub)+".h5", "w")
    ori_eegrdms = crosstime_corrs_cal(ori_subdata, modelrdm)
    f.create_dataset("ori", data=ori_eegrdms)
    pos_eegrdms = crosstime_corrs_cal(pos_subdata, modelrdm)
    f.create_dataset("pos", data=pos_eegrdms)
    f.close()

runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("corrs_crosst/Alpha_runtime.txt", runtime)