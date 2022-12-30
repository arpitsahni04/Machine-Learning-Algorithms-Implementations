import numpy as np
import sys
import csv
import math


def read_dataset(dataset):
    '''
    reads dataset, omits header and returns a np.array
    '''
    data = np.loadtxt(dataset, skiprows=1)
    with open(dataset, "r") as file:
        tsv_file = csv.reader(file, delimiter="\t")
        dataset = list(tsv_file)
        features = dataset[0]
    dataset = {"data": data, "features": features}
    return dataset


def entropy(y):
    '''
    calculates H(Y): purity or uniformity of a collection 
    of values:Lower the Entropy, more Pure
    0 when pure
    '''
    counts = {}
    for ele in y:
        if ele not in counts:
            counts[ele] = 1
        else:
            counts[ele] += 1
    # print("unique label values :",counts)
    H_y = 0
    for values, freq in counts.items():
        P_y = (freq/len(y))
        if P_y == 0:  # takes care of log(0) case
            continue
        H_y += -(P_y*math.log(P_y, 2))

    return H_y


def majority_vote(train_labels):
    '''
    Computes majority vote on the given column of dataset
    '''
    counts = {}
    vote = 0
    for ele in train_labels:
        if ele not in counts:
            counts[ele] = 1
        else:
            counts[ele] += 1
    if len(counts) == 0:
        return 1
    if len(counts) == 1:
        for ele in counts:
            return ele
    if counts[0] <= counts[1]:
        vote = 1
    return vote


def metrics(train_input, metrics_out):
    '''
    Computes entropy and majority vote 
    train_error and writes to a file
    '''
    dataset = read_dataset(train_input)
    train_labels = dataset["data"][:, -1]
    vote = majority_vote(train_labels)

    H_y = entropy(train_labels)
    predictions_train = vote * np.ones(len(train_labels))
    error_train = np.mean(predictions_train != train_labels, dtype=np.float32)

    with open(metrics_out, "w") as file:
        file.write("entropy: " + str(H_y) + "\n")
        file.write("error: " + str(error_train) + "\n")


if __name__ == '__main__':
    # read datasets
    args = sys.argv
    assert (len(args) == 3)
    train_input = args[1]
    metrics_out = args[2]

    metrics(train_input, metrics_out)
