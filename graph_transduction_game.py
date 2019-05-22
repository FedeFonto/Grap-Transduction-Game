import sklearn.metrics.pairwise as kernel
import numpy as np
import h5py

N_START = 0
N_END = 9
N_CLASS = N_END - N_START + 1

### load USPS dataset ###
PATH = "usps.h5"

with h5py.File(PATH, 'r') as hf:
    train = hf.get('test')
    X_tr = train.get('data')[:]
    Y_tr = train.get('target')[:]
    test = hf.get('train')
    X_te = test.get('data')[:]
    Y_te = test.get('target')[:]

### Preparation phase ###
data = []
strategy = []
label = []

for i in range(len(Y_tr)):
    if N_END + 1 > Y_tr[i] > N_START - 1:
        data.append(X_tr[i])
        zero = np.zeros(N_CLASS)
        label.append(Y_tr[i] - N_START)
        zero[Y_tr[i] - N_START] = 1
        strategy.append(zero)

n_train = len(data)

for i in range(len(Y_te)):
    if N_END + 1 > Y_te[i] > N_START - 1:
        data.append(X_te[i])
        uniform = np.zeros(N_CLASS)
        label.append(Y_te[i] - N_START)
        uniform.fill(1. / N_CLASS)
        strategy.append(uniform)

n_data = len(data)

similarity = kernel.rbf_kernel(data, data, gamma=1)
strategy = np.array(strategy)

### Function ###
count = 0
norm = 1

while norm > 0.01:
    q = np.dot(similarity, strategy)
    new_strategy = np.multiply(strategy, q)
    new_strategy /= new_strategy.sum(axis=1, keepdims=True)

    norm = np.linalg.norm(strategy - new_strategy)
    strategy = new_strategy
    count += 1
    print("End iteration {}".format(count))
    print(norm)

accuracy = np.sum(strategy.argmax(axis=1) == label) / float(n_data)
print("\nAccuracy: {}\n".format(accuracy))
