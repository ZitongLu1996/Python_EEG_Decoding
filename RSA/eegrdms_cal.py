import numpy as np
from neurora.rdm_cal import eegRDM
import h5py
import time

subs = ["201", "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "212", "213", "215", "216", "217", "218"]

nsubs = len(subs)

starttime = time.clock()

ori_data = np.zeros([16, nsubs, 40, 27, 500], dtype=np.float)
pos_data = np.zeros([16, nsubs, 40, 27, 500], dtype=np.float)

index = 0
for sub in subs:
    f = h5py.File("data_for_RSA/ERP/"+sub+".h5", "r")
    ori_subdata = np.array(f["ori"])
    pos_subdata = np.array(f["pos"])
    f.close()
    ori_data[:, index, :, :, :] = ori_subdata
    pos_data[:, index, :, :, :] = pos_subdata
    index = index + 1

ori_eegrdms = eegRDM(ori_data, sub_opt=1, time_opt=1, time_win=5, time_step=5)
f = h5py.File("eegrdms/ERP.h5", "w")
f.create_dataset("ori", data=ori_eegrdms)
pos_eegrdms = eegRDM(pos_data, sub_opt=1, time_opt=1, time_win=5, time_step=5)
f.create_dataset("pos", data=pos_eegrdms)
f.close()

runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("eegrdms/ERP_runtime.txt", runtime)


starttime = time.clock()

ori_data = np.zeros([16, nsubs, 40, 27, 500], dtype=np.float)
pos_data = np.zeros([16, nsubs, 40, 27, 500], dtype=np.float)

index = 0
for sub in subs:
    f = h5py.File("data_for_RSA/Alpha/"+sub+".h5", "r")
    ori_subdata = np.array(f["ori"])
    pos_subdata = np.array(f["pos"])
    f.close()
    ori_data[:, index, :, :, :] = ori_subdata
    pos_data[:, index, :, :, :] = pos_subdata
    index = index + 1

ori_eegrdms = eegRDM(ori_data, sub_opt=1, time_opt=1, time_win=5, time_step=5)
f = h5py.File("eegrdms/Alpha.h5", "w")
f.create_dataset("ori", data=ori_eegrdms)
pos_eegrdms = eegRDM(pos_data, sub_opt=1, time_opt=1, time_win=5, time_step=5)
f.create_dataset("pos", data=pos_eegrdms)
f.close()

runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("eegrdms/Alpha_runtime.txt", runtime)