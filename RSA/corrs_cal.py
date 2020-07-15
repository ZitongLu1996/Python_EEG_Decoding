import numpy as np
from neurora.corr_cal_by_rdm import rdms_corr
import h5py
import time

modelrdm = np.loadtxt("modelrdm/modelrdm.txt")

starttime = time.clock()
f = h5py.File("eegrdms/ERP.h5", "r")
ori_eegrdms = np.array(f["ori"])
pos_eegrdms = np.array(f["pos"])
f.close()
f = h5py.File("corrs/ERP.h5", "w")
corrs = rdms_corr(modelrdm, ori_eegrdms)
f.create_dataset("ori", data=corrs)
corrs = rdms_corr(modelrdm, pos_eegrdms)
f.create_dataset("pos", data=corrs)
f.close()
runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("corrs/ERP_runtime.txt", runtime)

starttime = time.clock()
f = h5py.File("eegrdms/Alpha.h5", "r")
ori_eegrdms = np.array(f["ori"])
pos_eegrdms = np.array(f["pos"])
f.close()
f = h5py.File("corrs/Alpha.h5", "w")
corrs = rdms_corr(modelrdm, ori_eegrdms)
f.create_dataset("ori", data=corrs)
corrs = rdms_corr(modelrdm, pos_eegrdms)
f.create_dataset("pos", data=corrs)
f.close()
runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("corrs/Alpha_runtime.txt", runtime)