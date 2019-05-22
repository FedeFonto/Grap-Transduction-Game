import random as rand
import sklearn.metrics.pairwise as kernel
import numpy as np
import h5py

N_START = 0
N_END = 4
N_CLASS = N_END - N_START + 1

def payoff(n):
    payoff_e = np.zeros(N_CLASS)
    payoff_m = 0
    # itero su tutti gli elementi gia etichettati
    for j in range(n_train):
        # calcolo il payoff extreme (4.3) seconda parte (moltiplico similarita per label)
        payoff_e[label[j]] += similarity[n, j]

        # calcolo il payoff mixed (4.4) seconda parte (moltiplico strategia per similarita)
        payoff_m += strategy[n, label[j]] * similarity[n, j]

    #itero su tutti gli elementi non etichettati
    for j in range(n_train, n_data):

        # calcolo il payoff extreme (4.3) prima parte (similarita[:,label j])
        payoff_e += strategy[j] * similarity[n, j]

        # calcolo il payoff mixed (4.4) prima parte (similarita[label x,label j])
        payoff_m += np.matmul(strategy[n], strategy[j]) * similarity[n, j]

    return payoff_e / payoff_m




### load USPS dataset ###
PATH = "usps.h5"

with h5py.File(PATH, 'r') as hf:
    train = hf.get('train')
    X_tr = train.get('data')[:]
    Y_tr = train.get('target')[:]
    test = hf.get('test')
    X_te = test.get('data')[:]
    Y_te = test.get('target')[:]

### Preparation phase ###
data = []
strategy = []
label = []
test = []

for i in range(len(Y_tr)):
    if Y_tr[i] > N_START -1 and Y_tr[i] < N_END + 1:
        data.append(X_tr[i])
        zero = np.zeros(N_CLASS)
        label.append(Y_tr[i] - N_START)
        zero[Y_tr[i] - N_START] = 1
        strategy.append(zero)

n_train = len(data)

for i in range(len(Y_te)):
    if Y_te[i] > N_START -1  and Y_te[i] < N_END + 1 :
        data.append(X_te[i])
        uniform = np.zeros(N_CLASS)
        test.append(Y_te[i] - N_START)
        label.append(rand.randint(0, N_CLASS - 1))
        uniform.fill(1. / N_CLASS)
        strategy.append(uniform)

n_data = len(data)

similarity = kernel.rbf_kernel(data, data, gamma=1)
sub = np.interp(similarity, (similarity.min(), similarity.max()), (0, 1))
similarity = sub
print similarity
strategy = np.array(strategy)

### Function ###
not_converged = 1
count = 0
norm = 1
while norm > 0.5 or not_converged != 0:
    not_converged = 0
    class_n = np.zeros(N_CLASS)
    for i in range(n_data):
        class_n[label[i]] += 1
    #print class_n

    new_strategy = np.array(strategy[0:n_data])
    new_label = np.array(label[0:n_data])

    for i in range(n_train, n_data):
        max_pos = -1
        max_value = 0
        pf = payoff(i)

        #print pf

        # calcolo la nuova strategia seguendo 4.9 per ogni h
        for h in range(N_CLASS):
            value = strategy[i][h] * pf[h]
            new_strategy[i][h] = value

            # se il valore e piu alto, seleziono la nuova etichetta
            if new_strategy[i][h] > max_value:
                max_value = new_strategy[i][h]
                max_pos = h

        new_strategy[i] = new_strategy[i] / np.sum(new_strategy[i])

        if max_pos != label[i]:
            not_converged += 1

        new_label[i] = max_pos

        #print new_strategy[i]
        #print max_pos, label[i], test[i-n_train]

    norm = np.linalg.norm(strategy - new_strategy)
    strategy = new_strategy
    label = new_label
    count += 1
    print "End iteration {}, label change: {}".format(count, not_converged)
    print norm
    accuracy = 0.
    for i in range(n_train, n_data):
        if label[i] == test[i-n_train]:
            accuracy += 1
    accuracy = accuracy / (n_data - n_train)
    print "Accuracy: {}\n".format(accuracy)

