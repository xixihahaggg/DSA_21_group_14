import numpy as np
from pprint import pprint
from collections import Counter
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class TreeNode(object):
    """The basic node of tree structure"""

    def __init__(self, name, depth):
        super(TreeNode, self).__init__()
        self.name = name
        self.depth = depth
        self.left = None
        self.right = None

    def __repr__(self):
        return 'Variable {}, depth is {}'.format(self.name, self.depth)


def generating_data(num_samples, dim):
    """
    :param num_samples:
    :param dim:
    :return: Binary Data and Label
    """
    # X_1 = 1 with probability 1/2, X_1 = 0 with probability 1/2
    # For i = 2, . . . , k, X_i = X_i−1 with probability 3/4, and X_i = 1 − X_i−1 with probability 1/4.

    X = np.random.binomial(1., 0.5, num_samples).reshape(num_samples, -1)
    for i in range(1, dim):
        # if 1 maintain, if 0 reverse, xnor
        prob = np.random.binomial(1., .75, num_samples)
        x = np.logical_not(np.logical_xor(X[:, -1], prob)).reshape(num_samples, -1).astype(np.float)
        X = np.append(X, x, axis=-1)
    # In other words, if the ‘weighted average’ of X 2 , . . . X k tilts high, Y will agree with X 1 ; if the
    # weighted average of X 2 , . . . , X k tilts low, Y will disagree with X 1 . Take the weights to be defined by w
    # i = 0.9 i /(0.9 2 + 0.9 3 + . . . + 0.9 k ).
    x = [np.power(0.9, i) for i in range(2, dim + 1)]
    weight = np.array(x / sum(x)).reshape(-1, 1)
    y = X[:, 1:] @ weight
    y = np.where(y > 0.5, X[:, 0].reshape(-1, 1), 1 - X[:, 0].reshape(-1, 1))
    return X, y


def entropy(v):
    if np.all(v == 0) or np.all(v == 1):
        return .0
    x = np.mean(v)
    return -(x * np.log2(x) + (1 - x) * np.log2(1 - x))


def linear_combination(x, y, theta):
    return theta * x + (1 - theta) * y


def find_max_info_gain(X, y):
    """
    An implementation of ID3 algorithm
    :param X: Data
    :param y: Label
    :return: the index of X that has the maximum IG
    """
    num_samples = X.shape[0]

    H = entropy(y)
    if H == 0:
        return None

    gains = []
    # TODO split X according to x_i and y
    for i in range(X.shape[1]):
        # find the y vector where x_i = 1
        x_1 = y[np.where(X[:, i] == 1.)]
        # find the y vector where x_i = 0
        x_0 = y[np.where(X[:, i] == 0.)]
        entropy_x = linear_combination(entropy(x_1), entropy(x_0), len(x_1) / num_samples)
        gain = H - entropy_x
        gains.append(gain)

    if max(gains) <= 0.0:
        return None
    else:
        return gains.index(max(gains))


def find_max_gain_ratio(X, y):
    """
    An implementation of C4.5 algorithm
    :param X: Data
    :param y: Label
    :return: the index of X that has the maximum information gain ratio
    """
    num_samples = X.shape[0]
    H = entropy(y)
    if H == 0:
        return None
    gains = []
    # TODO split X according to x_i and y
    for i in range(X.shape[1]):
        # find the y vector where x_i = 1
        x_1 = y[np.where(X[:, i] == 1.)]
        # find the y vector where x_i = 0
        x_0 = y[np.where(X[:, i] == 0.)]
        n1 = len(x_1) / (len(x_1) + len(x_0))
        split_info = -linear_combination(np.log2(n1), np.log2(1 - n1), n1)
        entropy_x = linear_combination(entropy(x_1), entropy(x_0), len(x_1) / num_samples)
        gain = H - entropy_x
        gains.append(gain / split_info)
    if max(gains) == 0.0:
        return None
    else:
        return gains.index(max(gains))


def split_data(X, y, index):
    """
    :param X: Data
    :param y: Label
    :param index: variable index
    :return: two dataset split based the variable X_i (X_1,y_1),(X_0,y_0)
    """
    x_1 = np.where(X[:, index] == 1)[0]
    x_0 = np.where(X[:, index] == 0)[0]
    # Set the value of x_i as nan, in order to skip in finer level split
    # X[:, index] = np.nan

    y_1, y_0 = y[x_1], y[x_0]
    X_1, X_0 = X[x_1, :], X[x_0, :]
    return (X_1, y_1), (X_0, y_0)


def build_Decision_Tree(X, y, maximum_depth, minimum_sample_size=None, split_method="IG", depth=0, error=0):
    """
    :param X: Data
    :param y: Label
    :param depth: Depth of the tree node
    :param error: Number of error when building the tree
    :param split_method: 'IG' for ID3 based algorithm, else C4.5.
    :param minimum_sample_size: Number of minimum sample size for pruning the tree
    :return: Root of decision tree
    """
    # if depth is greater than the dim, it's not separable, in other word, if depth > maximum depth, the maximum IG is 0
    if depth >= maximum_depth:
        decision = 1. if np.mean(y) >= 0.5 else 0.
        error += np.abs(decision * len(y) - np.sum(y))
        return decision, error

    if isinstance(minimum_sample_size, int):
        if len(y) < minimum_sample_size:
            decision = 1. if np.mean(y) >= 0.5 else 0.
            error += np.abs(decision * len(y) - np.sum(y))
            return decision, error

    if split_method == "IG":
        index_root = find_max_info_gain(X, y)
    else:
        index_root = find_max_gain_ratio(X, y)
    # max gain is 0, it has noise in label
    if index_root is None:
        if depth == 0:
            print("IGs for variable are 0, could not use ID3")
            return None, error
        decision = 1. if np.mean(y) >= 0.5 else 0.
        error += np.abs(decision * len(y) - np.sum(y))
        return decision, error

    node = TreeNode(index_root, depth=depth + 1)
    # split by the dim index
    (X_1, y_1), (X_0, y_0) = split_data(X, y, index_root)
    node.right, error = build_Decision_Tree(X_1, y_1, minimum_sample_size=minimum_sample_size,
                                            maximum_depth=maximum_depth, depth=depth + 1, error=error)
    node.left, error = build_Decision_Tree(X_0, y_0, minimum_sample_size=minimum_sample_size,
                                           maximum_depth=maximum_depth, depth=depth + 1, error=error)

    return node, error


def traverse_by_level(root, debug=False):
    q = [root]
    tree_node = []
    while len(q) > 0:
        # print(q)
        node = q[0]
        del q[0]
        if isinstance(node, TreeNode):
            if debug:
                print("Node {}".format(node))
            tree_node.append(node.name)
            q.append(node.left)
            q.append(node.right)
        elif isinstance(node, float):
            if debug:
                print("Decision {}".format(node))
    return tree_node


def predict(root, X, y):
    if not isinstance(root, TreeNode):
        return root
    else:
        if X[root.name] == 1:
            return predict(root.right, X, y)
        elif X[root.name] == 0:
            return predict(root.left, X, y)


def cal_error_rate(root, X, y):
    cnt = 0
    num_samples = X.shape[0]
    for i in range(num_samples):
        prediction = predict(root, X[i, :], y[i, :])
        if prediction != y[i, :]:
            cnt += 1
    return cnt / num_samples


def plot_graph(dic):
    n, training_time, testing_time, acc =[], [], [], []
    for k, v in dic.items():
        n.append(k)
        training_time.append(v["Training Time"])
        testing_time.append(v["Testing Time"])
        acc.append(v["Testing Accuracy"])

    plt.title('Relation between number of samples and training time')
    plt.xlabel('number of samples')
    plt.ylabel('Time')
    plt.plot(n, training_time, 'ro-')
    plt.show()

    plt.title('Relation between number of samples and testing time')
    plt.xlabel('number of samples')
    plt.ylabel('Time')
    plt.plot(n, testing_time, 'ro-')
    plt.show()

    plt.title('Relation between number of samples and accuracy')
    plt.xlabel('number of samples')
    plt.ylabel('Accuracy')
    plt.plot(n, acc, 'ro-')
    plt.show()


def cal_underlying_error(num_samples, dim, epochs, maximum_depth=None, minimum_sample_size=None, split_method="IG"):
    train_errors = []
    test_errors = []
    training_time = []
    testing_time = []
    dic = {}
    if maximum_depth is None:
        maximum_depth = dim
    for i in range(epochs):
        X_train, y_train = generating_data(num_samples, dim)
        X_test, y_test = generating_data(num_samples, dim)
        if np.mean(y) == .0 or np.mean(y) == 1.:
            print("Bad dataset")
        train_start = time.time()
        tree, train_error = build_Decision_Tree(X_train, y_train, split_method=split_method,
                                                minimum_sample_size=minimum_sample_size,
                                                maximum_depth=maximum_depth)
        train_end = time.time()
        train_errors.append(train_error / len(y_train))
        test_start = time.time()
        test_error = cal_error_rate(tree, X_test, y_test)
        test_end = time.time()
        test_errors.append(test_error)
        training_time.append(train_end - train_start)
        testing_time.append(test_end-test_start)
        if isinstance(tree, TreeNode):
            if tree.name not in dic.keys():
                dic[tree.name] = 0
            dic[tree.name] += 1
    return dic, np.mean(train_errors), np.mean(test_errors), np.mean(training_time), np.mean(testing_time)


def load_iris_data():
    d = {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}
    df = pd.read_csv("./data/Iris.csv").iloc[:, 1:]
    df["Species"] = [d[i] for i in df["Species"].to_list()]
    data = df.to_numpy()
    return data[:, :-1], data[:, -1]


def load_wine_data():
    # d = {"Iris-setosa": 1, "Iris-versicolor": 2, "Iris-virginica": 3}
    d = {"white": 1, "red": 0}
    df = pd.read_csv("./data/winequalityN.csv")
    df["type"] = [d[i] for i in df["type"].to_list()]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    data = df.to_numpy()

    return data[:, 1:], data[:, 0]


def load_birds_data():
    train = pd.read_csv("./data/train.csv")
    train = train.drop(columns=["genus"])
    species = {specie: index for index, specie in enumerate(sorted(list(set(train["species"].to_list()))))}
    test = pd.read_csv("./data/test.csv")
    test = test.drop(columns=["genus"])

    train["species"] = [species[i] for i in train["species"].to_list()]
    train = train.apply(pd.to_numeric, errors='coerce')
    train = train.dropna()
    train = train.to_numpy()

    train_x = train[:, :-1]
    train_y = train[:, -1]
    test["species"] = [species[i] for i in test["species"].to_list()]
    test = test.apply(pd.to_numeric, errors='coerce')
    test = test.dropna()
    test = test.to_numpy()

    test_x = test[:, :-1]
    test_y = test[:, -1]

    return train_x, train_y, test_x, test_y


def iris_exp():
    np.random.seed(7)
    data, target = load_iris_data()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(Xtrain, Ytrain)
    print("For ID3 based method, training results is {}, testing results is {}"
          .format(clf.score(Xtrain, Ytrain), clf.score(Xtest, Ytest)))

    clf = DecisionTreeClassifier(criterion="gini")
    clf = clf.fit(Xtrain, Ytrain)
    print("For CART based method, training results is {}, testing results is {}"
          .format(clf.score(Xtrain, Ytrain), clf.score(Xtest, Ytest)))


def wine_exp():
    data, target = load_wine_data()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, target, test_size=0.3)
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(Xtrain, Ytrain)
    print("For ID3 based method, training results is {}, testing results is {}"
          .format(clf.score(Xtrain, Ytrain), clf.score(Xtest, Ytest)))

    clf = DecisionTreeClassifier(criterion="gini")
    clf = clf.fit(Xtrain, Ytrain)
    print("For CART based method, training results is {}, testing results is {}"
          .format(clf.score(Xtrain, Ytrain), clf.score(Xtest, Ytest)))


def plot_graph1(dic):
    n, time_entropy, acc_entropy, time_gini, acc_gini = [], [], [], [], []
    for k, v in dic.items():
        n.append(k)
        time_entropy.append(v["Entropy"]["Time"])
        acc_entropy.append(v["Entropy"]["Acc"])
        time_gini.append(v["Gini"]["Time"])
        acc_gini.append(v["Gini"]["Acc"])

    plt.title('Relation between number of samples and training&testing time, Entropy')
    plt.xlabel('number of estimators')
    plt.ylabel('Time')
    plt.plot(n, time_entropy, 'ro-')
    plt.show()

    plt.title('Relation between number of samples and accuracy, Entropy')
    plt.xlabel('number of estimators')
    plt.ylabel('Accuracy')
    plt.plot(n, acc_entropy, 'ro-')
    plt.show()

    plt.title('Relation between number of samples and training&testing time, Gini')
    plt.xlabel('number of estimators')
    plt.ylabel('Time')
    plt.plot(n, time_gini, 'ro-')
    plt.show()

    plt.title('Relation between number of samples and accuracy, Gini')
    plt.xlabel('number of estimators')
    plt.ylabel('Accuracy')
    plt.plot(n, acc_gini, 'ro-')
    plt.show()


def bird_song_exp():
    train_x, train_y, test_x, test_y = load_birds_data()
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(train_x, train_y)
    print("For ID3 based method, training results is {}, testing results is {}"
          .format(clf.score(train_x, train_y), clf.score(test_x, test_y)))

    clf = DecisionTreeClassifier(criterion="gini")
    clf = clf.fit(train_x, train_y)
    print("For CART based method, training results is {}, testing results is {}"
          .format(clf.score(train_x, train_y), clf.score(test_x, test_y)))
    dic = {}

    for n in tqdm(range(1, 101)):
        start = time.time()
        rfc = RandomForestClassifier(criterion="gini", random_state=0, n_estimators=n)
        rfc = rfc.fit(train_x, train_y)
        score_gini = rfc.score(test_x, test_y)
        end = time.time()
        time_gini = end - start

        start = time.time()
        rfc = RandomForestClassifier(criterion="entropy", random_state=0, n_estimators=n)
        rfc = rfc.fit(train_x, train_y)
        score_entropy = rfc.score(test_x, test_y)
        end = time.time()
        time_entropy = end - start

        dic[n] = {"Gini": {"Time": time_gini, "Acc": score_gini},
                  "Entropy": {"Time": time_entropy, "Acc": score_entropy}}

    pprint(dic)
    plot_graph1(dic)


if __name__ == '__main__':
    """
    Part 1, numerical experiment based on my implementation
    """
    dic = {}

    for n in tqdm(range(10000, 200001, 10000)):
        _, _, error, training_time, testing_time = cal_underlying_error(num_samples=n, dim=21, epochs=10)
        dic[n] = {"Testing Accuracy": 1 - error, "Training Time": training_time, "Testing Time":testing_time}
    pprint(dic)
    plot_graph(dic)

    for n in tqdm(range(10000, 200001, 10000)):
        _, _, error, training_time, testing_time = cal_underlying_error(num_samples=n, dim=21, epochs=10, split_method="IG")
        dic[n] = {"Testing Accuracy": 1 - error, "Training Time": training_time, "Testing Time": testing_time}

    pprint(dic)
    plot_graph(dic)

    """
    Part 2, case studies based on sklearn toolkit
    """

    np.random.seed(0)
    iris_exp()
    wine_exp()
    bird_song_exp()
