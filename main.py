import os

import numpy as np
import torch


def read_data() -> tuple:
    data = []
    labels = np.array([])
    for i in range(1, 4):
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


def train_test_split(data, labels, num) -> tuple:
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
    data_size = shuffle_idx.size(0)
    percent80 = int(data_size * 0.8)
    percent60 = int(data_size * 0.6)
    percent40 = int(data_size * 0.4)
    percent20 = int(data_size * 0.2)

    # Slicing into X_train, X_test, y_train, y_test
    if num == 1:
        X_train, X_test = X[shuffle_idx[:percent80]], X[shuffle_idx[percent80:]]
        y_train, y_test = y[shuffle_idx[:percent80]], y[shuffle_idx[percent80:]]

    elif num == 2:
        X_train, X_test = X[torch.tensor(list(shuffle_idx[:percent60]) + list(shuffle_idx[percent80:]))], X[shuffle_idx[percent60:percent80]]
        y_train, y_test = y[torch.tensor(list(shuffle_idx[:percent60]) + list(shuffle_idx[percent80:]))], y[shuffle_idx[percent60:percent80]]

    elif num == 3:
        X_train, X_test = X[torch.tensor(list(shuffle_idx[:percent40]) + list(shuffle_idx[percent60:]))], X[shuffle_idx[percent40:percent60]]
        y_train, y_test = y[torch.tensor(list(shuffle_idx[:percent40]) + list(shuffle_idx[percent60:]))], y[shuffle_idx[percent40:percent60]]

    elif num == 4:
        X_train, X_test = X[torch.tensor(list(shuffle_idx[:percent20]) + list(shuffle_idx[percent40:]))], X[shuffle_idx[percent20:percent40]]
        y_train, y_test = y[torch.tensor(list(shuffle_idx[:percent20]) + list(shuffle_idx[percent40:]))], y[shuffle_idx[percent20:percent40]]

    elif num == 5:
        X_train, X_test = X[shuffle_idx[percent20:]], X[shuffle_idx[:percent20]]
        y_train, y_test = y[shuffle_idx[percent20:]], y[shuffle_idx[:percent20]]

    return X_train, X_test, y_train, y_test


def run_adaline():
    data, labels = read_data()
    for i in range(1, 6):
        print(i)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, i)


if __name__ == '__main__':
    run_adaline()
