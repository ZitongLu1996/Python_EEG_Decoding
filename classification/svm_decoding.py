import numpy as np
import scipy.io as sio
import random
import h5py
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

subs = ["201", "202", "203", "204", "205", "206", "207", "208", "209",
        "210", "212", "213", "215", "216", "217", "218"]

runtime = np.zeros([len(subs)], dtype=np.float)
subindex = 0
f = h5py.File("svm_results/ERP_ori.h5", "w")
for sub in subs:
    starttime = time.clock()
    fdata = h5py.File("data_for_classification/ERP/"+sub+".h5", "r")
    data = np.array(fdata["ori"])
    fdata.close()
    print(data.shape)
    acc = np.zeros([100, 100, 100, 3], dtype=np.float)
    for k in range(100):
        print(sub, k)
        index_trials = np.array(range(40))
        shuffle = np.random.permutation(index_trials)
        newdata = data[:, shuffle[:39]]
        block_data = np.zeros([3, 16, 27, 100], dtype=np.float)
        for i in range(3):
            block_data[i] = np.average(newdata[:, i*13:i*13+13], axis=1)
        y_train = np.zeros([2*16], dtype=np.int)
        for i in range(2):
            for j in range(16):
                y_train[i*16+j] = j
        y_test = np.zeros([16], dtype=np.int)
        for i in range(16):
            y_test[i] = i
        for i in range(3):
            x_test = block_data[i]
            x_train = np.zeros([2, 16, 27, 100], dtype=np.float)
            index = 0
            for j in range(3):
                if j != i:
                    x_train[index] = block_data[j]
                    index = index + 1
            x_train = np.reshape(x_train, [2*16, 27, 100])
            for t in range(100):
                x_train_t = x_train[:, :, t]
                x_test_t = x_test[:, :, t]
                svm = SVC(kernel='linear', decision_function_shape='ovr')
                svm.fit(x_train_t, y_train)
                y_pred = svm.predict(x_test_t)
                acc[k, t, t, i] = accuracy_score(y_test, y_pred)
                for tt in range(99):
                    if tt < t:
                        x_test_tt = x_test[:, :, tt]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt, i] = accuracy_score(y_test, y_pred)
                    if tt >= t:
                        x_test_tt = x_test[:, :, tt+1]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt+1, i] = accuracy_score(y_test, y_pred)
    f.create_dataset(sub, data=np.average(acc, axis=(0, 3)))
runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("svm_results/ERP_ori_runtime.txt", runtime)


runtime = np.zeros([len(subs)], dtype=np.float)
subindex = 0
f = h5py.File("svm_results/ERP_pos.h5", "w")
for sub in subs:
    starttime = time.clock()
    fdata = h5py.File("data_for_classification/ERP/"+sub+".h5", "r")
    data = np.array(fdata["pos"])
    fdata.close()
    print(data.shape)
    acc = np.zeros([100, 100, 100, 3], dtype=np.float)
    for k in range(100):
        print(sub, k)
        index_trials = np.array(range(40))
        shuffle = np.random.permutation(index_trials)
        newdata = data[:, shuffle[:39]]
        block_data = np.zeros([3, 16, 27, 100], dtype=np.float)
        for i in range(3):
            block_data[i] = np.average(newdata[:, i*13:i*13+13], axis=1)
        y_train = np.zeros([2*16], dtype=np.int)
        for i in range(2):
            for j in range(16):
                y_train[i*16+j] = j
        y_test = np.zeros([16], dtype=np.int)
        for i in range(16):
            y_test[i] = i
        for i in range(3):
            x_test = block_data[i]
            x_train = np.zeros([2, 16, 27, 100], dtype=np.float)
            index = 0
            for j in range(3):
                if j != i:
                    x_train[index] = block_data[j]
                    index = index + 1
            x_train = np.reshape(x_train, [2*16, 27, 100])
            for t in range(100):
                x_train_t = x_train[:, :, t]
                x_test_t = x_test[:, :, t]
                svm = SVC(kernel='linear', decision_function_shape='ovr')
                svm.fit(x_train_t, y_train)
                y_pred = svm.predict(x_test_t)
                acc[k, t, t, i] = accuracy_score(y_test, y_pred)
                for tt in range(99):
                    if tt < t:
                        x_test_tt = x_test[:, :, tt]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt, i] = accuracy_score(y_test, y_pred)
                    if tt >= t:
                        x_test_tt = x_test[:, :, tt+1]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt+1, i] = accuracy_score(y_test, y_pred)
    f.create_dataset(sub, data=np.average(acc, axis=(0, 3)))
runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("svm_results/ERP_pos_runtime.txt", runtime)


runtime = np.zeros([len(subs)], dtype=np.float)
subindex = 0
f = h5py.File("svm_results/Alpha_ori.h5", "w")
for sub in subs:
    starttime = time.clock()
    fdata = h5py.File("data_for_classification/Alpha/"+sub+".h5", "r")
    data = np.array(fdata["ori"])
    fdata.close()
    print(data.shape)
    acc = np.zeros([100, 100, 100, 3], dtype=np.float)
    for k in range(100):
        print(sub, k)
        index_trials = np.array(range(40))
        shuffle = np.random.permutation(index_trials)
        newdata = data[:, shuffle[:39]]
        block_data = np.zeros([3, 16, 27, 100], dtype=np.float)
        for i in range(3):
            block_data[i] = np.average(newdata[:, i*13:i*13+13], axis=1)
        y_train = np.zeros([2*16], dtype=np.int)
        for i in range(2):
            for j in range(16):
                y_train[i*16+j] = j
        y_test = np.zeros([16], dtype=np.int)
        for i in range(16):
            y_test[i] = i
        for i in range(3):
            x_test = block_data[i]
            x_train = np.zeros([2, 16, 27, 100], dtype=np.float)
            index = 0
            for j in range(3):
                if j != i:
                    x_train[index] = block_data[j]
                    index = index + 1
            x_train = np.reshape(x_train, [2*16, 27, 100])
            for t in range(100):
                x_train_t = x_train[:, :, t]
                x_test_t = x_test[:, :, t]
                svm = SVC(kernel='linear', decision_function_shape='ovr')
                svm.fit(x_train_t, y_train)
                y_pred = svm.predict(x_test_t)
                acc[k, t, t, i] = accuracy_score(y_test, y_pred)
                for tt in range(99):
                    if tt < t:
                        x_test_tt = x_test[:, :, tt]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt, i] = accuracy_score(y_test, y_pred)
                    if tt >= t:
                        x_test_tt = x_test[:, :, tt+1]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt+1, i] = accuracy_score(y_test, y_pred)
    f.create_dataset(sub, data=np.average(acc, axis=(0, 3)))
runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("svm_results/Alpha_ori_runtime.txt", runtime)


runtime = np.zeros([len(subs)], dtype=np.float)
subindex = 0
f = h5py.File("svm_results/Alpha_pos.h5", "w")
for sub in subs:
    starttime = time.clock()
    fdata = h5py.File("data_for_classification/Alpha/"+sub+".h5", "r")
    data = np.array(fdata["pos"])
    fdata.close()
    print(data.shape)
    acc = np.zeros([100, 100, 100, 3], dtype=np.float)
    for k in range(100):
        print(sub, k)
        index_trials = np.array(range(40))
        shuffle = np.random.permutation(index_trials)
        newdata = data[:, shuffle[:39]]
        block_data = np.zeros([3, 16, 27, 100], dtype=np.float)
        for i in range(3):
            block_data[i] = np.average(newdata[:, i*13:i*13+13], axis=1)
        y_train = np.zeros([2*16], dtype=np.int)
        for i in range(2):
            for j in range(16):
                y_train[i*16+j] = j
        y_test = np.zeros([16], dtype=np.int)
        for i in range(16):
            y_test[i] = i
        for i in range(3):
            x_test = block_data[i]
            x_train = np.zeros([2, 16, 27, 100], dtype=np.float)
            index = 0
            for j in range(3):
                if j != i:
                    x_train[index] = block_data[j]
                    index = index + 1
            x_train = np.reshape(x_train, [2*16, 27, 100])
            for t in range(100):
                x_train_t = x_train[:, :, t]
                x_test_t = x_test[:, :, t]
                svm = SVC(kernel='linear', decision_function_shape='ovr')
                svm.fit(x_train_t, y_train)
                y_pred = svm.predict(x_test_t)
                acc[k, t, t, i] = accuracy_score(y_test, y_pred)
                for tt in range(99):
                    if tt < t:
                        x_test_tt = x_test[:, :, tt]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt, i] = accuracy_score(y_test, y_pred)
                    if tt >= t:
                        x_test_tt = x_test[:, :, tt+1]
                        y_pred = svm.predict(x_test_tt)
                        acc[k, t, tt+1, i] = accuracy_score(y_test, y_pred)
    f.create_dataset(sub, data=np.average(acc, axis=(0, 3)))
runtime = np.zeros([1], dtype=np.float)
runtime[0] = str(time.clock() - starttime)
np.savetxt("svm_results/Alpha_pos_runtime.txt", runtime)