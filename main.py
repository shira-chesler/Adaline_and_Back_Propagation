import numpy as np
import torch

EPSILLON = 1e-5


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
    y = torch.tensor(np.array(labels).astype(int), dtype=torch.int)

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

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Slicing into X_train, X_test, y_train, y_test
    if num == 1:
        X_train, X_test = X[shuffle_idx[:percent80]], X[shuffle_idx[percent80:]]
        y_train, y_test = y[shuffle_idx[:percent80]], y[shuffle_idx[percent80:]]

    elif num == 2:
        X_train, X_test = X[torch.tensor(list(shuffle_idx[:percent60]) + list(shuffle_idx[percent80:]))], \
                          X[shuffle_idx[percent60:percent80]]
        y_train, y_test = y[torch.tensor(list(shuffle_idx[:percent60]) + list(shuffle_idx[percent80:]))], y[
            shuffle_idx[percent60:percent80]]

    elif num == 3:
        X_train, X_test = X[torch.tensor(list(shuffle_idx[:percent40]) + list(shuffle_idx[percent60:]))], \
                          X[shuffle_idx[percent40:percent60]]
        y_train, y_test = y[torch.tensor(list(shuffle_idx[:percent40]) + list(shuffle_idx[percent60:]))], y[
            shuffle_idx[percent40:percent60]]

    elif num == 4:
        X_train, X_test = X[torch.tensor(list(shuffle_idx[:percent20]) + list(shuffle_idx[percent40:]))], \
                          X[shuffle_idx[percent20:percent40]]
        y_train, y_test = y[torch.tensor(list(shuffle_idx[:percent20]) + list(shuffle_idx[percent40:]))], y[
            shuffle_idx[percent20:percent40]]

    elif num == 5:
        X_train, X_test = X[shuffle_idx[percent20:]], X[shuffle_idx[:percent20]]
        y_train, y_test = y[shuffle_idx[percent20:]], y[shuffle_idx[:percent20]]

    return X_train, X_test, y_train, y_test


def adjust_weights(X_train, y_train) -> np.array:
    # weights = np.empty(100)
    # weights.fill(0.1)
    weights = np.random.uniform(low=0, high=0.1, size=100)
    # print(weights)
    bias = 0
    learning_rate = 0.015
    max_weight_change = 1
    while max_weight_change > EPSILLON:
        max_weight_change = 0
        for i in range(0, len(X_train)):
            # print(i)
            y_in = bias + (X_train[i] @ weights).sum()
            bias += learning_rate * (y_train[i] - y_in)
            for j in range(0, 100):
                add_weight = learning_rate * (y_train[i] - y_in) * X_train[i, j]
                weights[j] += add_weight
                if add_weight > max_weight_change:
                    max_weight_change = add_weight
                    # print("iteration: ", i, "new max: ", max_weight_change)
        # print(weights)
    return bias, weights


def predict(X_test, weights, bias, label1, label2):
    y_pred = np.dot(X_test, weights) + bias
    # Apply the step function to convert y_pred to the given labels
    y_pred_labels = np.where(y_pred >= 0, label2, label1)
    return y_pred_labels


def compute_accuracy(y_test, y_pred):
    assert len(y_test) == len(y_pred)
    num_correct = np.sum(y_test == y_pred)
    accuracy = num_correct / len(y_test)
    return accuracy


def run_adaline(first, second):
    data, labels = read_data()

    classify_mem_vs_bet_data = [x for i, x in enumerate(data) if (labels[i] == 1 or labels[i] == 3)]
    classify_mem_vs_bet_labels = [x for x in labels if (x == 1 or x == 3)]

    classify_lamed_vs_bet_data = [x for i, x in enumerate(data) if (labels[i] == 1 or labels[i] == 2)]
    classify_lamed_vs_bet_labels = [x for x in labels if (x == 1 or x == 2)]

    classify_lamed_vs_mem_data = [x for i, x in enumerate(data) if (labels[i] == 2 or labels[i] == 3)]
    classify_lamed_vs_mem_labels = [x for x in labels if (x == 2 or x == 3)]

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    scores = []

    first_label = 0
    second_label = 0

    for i in range(1, 6):
        if first == "b" and second == "m":
            X_train, X_test, y_train, y_test = train_test_split(classify_mem_vs_bet_data, classify_mem_vs_bet_labels, i)
            first_label = 1
            second_label = 3
        if first == "b" and second == "l":
            X_train, X_test, y_train, y_test = train_test_split(classify_lamed_vs_bet_data,
                                                                classify_lamed_vs_bet_labels, i)
            first_label = 1
            second_label = 2
        if first == "l" and second == "m":
            X_train, X_test, y_train, y_test = train_test_split(classify_lamed_vs_mem_data,
                                                                classify_lamed_vs_mem_labels, i)
            first_label = 2
            second_label = 3
        X_train = np.array([t.numpy() for t in X_train])
        y_train = y_train.cpu().detach().numpy()

        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        bias, weights = adjust_weights(X_train, y_train)
        print("weights:", weights)
        y_pred_labels = predict(X_test, weights, bias, first_label, second_label)
        accuracy = compute_accuracy(y_test, y_pred_labels)
        print(f'Accuracy for fold {i}: {accuracy:.2f}')
        scores.append(accuracy)
    average_accuracy = np.mean(scores)
    print(f'Average accuracy: {average_accuracy:.2f}')


if __name__ == '__main__':
    run_adaline('l', 'm')
