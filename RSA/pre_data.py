import numpy as np
import scipy.io as sio
import h5py

subs = ["201", "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "212", "213", "215", "216", "217", "218"]

for sub in subs:
    data = sio.loadmat("../data/ERP" + sub + ".mat")["filtData"][:, :, 250:]
    print(data.shape)
    # data.shape: n_trials, n_channels, n_times

    ori_label = np.loadtxt("../labels/ori_" + sub + ".txt")[:, 1]
    pos_label = np.loadtxt("../labels/pos_" + sub + ".txt")[:, 1]

    ori_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)
    pos_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)

    ori_labelindex = np.zeros([16], dtype=np.int)
    pos_labelindex = np.zeros([16], dtype=np.int)

    for i in range(640):
       label = int(ori_label[i])
       ori_subdata[label, ori_labelindex[label]] = data[i]
       ori_labelindex[label] = ori_labelindex[label] + 1
       label = int(pos_label[i])
       pos_subdata[label, pos_labelindex[label]] = data[i]
       pos_labelindex[label] = pos_labelindex[label] + 1

    f = h5py.File("data_for_RSA/ERP/"+sub+".h5", "w")
    f.create_dataset("ori", data=ori_subdata)
    f.create_dataset("pos", data=pos_subdata)
    f.close()


    data = sio.loadmat("../data/Alpha" + sub + ".mat")["filtData"][:, :, 250:]
    print(data.shape)
    # data.shape: n_trials, n_channels, n_times

    ori_label = np.loadtxt("../labels/ori_" + sub + ".txt")[:, 1]
    pos_label = np.loadtxt("../labels/pos_" + sub + ".txt")[:, 1]

    ori_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)
    pos_subdata = np.zeros([16, 40, 27, 500], dtype=np.float)

    ori_labelindex = np.zeros([16], dtype=np.int)
    pos_labelindex = np.zeros([16], dtype=np.int)

    for i in range(640):
       label = int(ori_label[i])
       ori_subdata[label, ori_labelindex[label]] = data[i]
       ori_labelindex[label] = ori_labelindex[label] + 1
       label = int(pos_label[i])
       pos_subdata[label, pos_labelindex[label]] = data[i]
       pos_labelindex[label] = pos_labelindex[label] + 1

    f = h5py.File("data_for_RSA/Alpha/" + sub + ".h5", "w")
    f.create_dataset("ori", data=ori_subdata)
    f.create_dataset("pos", data=pos_subdata)
    f.close()