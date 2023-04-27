import os

import numpy as np
import torch


def read_data() -> tuple:
    data = []
    labels = np.array([])
    for i in range(1,2):
        path = "C:\\Ariel codes\\NN\\Adaline\\" + str(i) + ".txt"
        with open(path) as f:
            idx = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if line == '\n':
                    continue
                line = line.replace('(', '')
                line = line.replace(')', '')
                line_data_str = line.split(',')
                line_data_int = list(map(int, line_data_str))
                labels = np.append(labels, line_data_int[0])
                del line_data_int[0]
                data.append(np.array(line_data_int))
                idx += 1
            f.close()
    return data, labels


def train_test_split(data, labels) -> tuple:

    # Check no error occurred during reading
    if len(data) != len(labels):
        print("ERROR")
        exit(1)
    data_length = len(labels)

    # Converting to torch type
    X = torch.tensor(np.array(data))
    y = torch.tensor(labels, dtype=torch.int)

    # Creates a random shuffle
    torch.manual_seed(42)
    shuffle_idx = torch.randperm(data_length, dtype=torch.long)

    # Organizes the data according to shuffle
    X, y = X[shuffle_idx], y[shuffle_idx]

    # Size of 80% of the data
    percent80 = int(shuffle_idx.size(0) * 0.8)

    # Slicing into X_train, X_test, y_train, y_test
    X_train, X_test = X[shuffle_idx[:percent80]], X[shuffle_idx[percent80:]]
    y_train, y_test = y[shuffle_idx[:percent80]], y[shuffle_idx[percent80:]]

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    data, labels = read_data()
    X_train, X_test, y_train, y_test = train_test_split(data, labels)
