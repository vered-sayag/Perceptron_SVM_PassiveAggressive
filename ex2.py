# importing the libraries
import numpy as np
import sys

def preperation(X):
    X = encoder(X)
    X = np.array(X)
    return X


def split_data(X, Y, test):
    #    X = X.rename_axis('ID')
    #   X = X.values
    Y = Y.rename_axis('ID')
    Y = Y.values
    X_size = X.shape
    triningX = []
    testX = []
    for i in range(10):
        if i != test:
            for j in range(int((i * X_size[0]) / 10), int(((i + 1) * X_size[0]) / 10)):
                triningX.append(X[j])

    for j in range(int(test * X_size[0] / 10), int((test + 1) * X_size[0] / 10)):
        testX.append(X[j])

    Y_size = Y.shape
    triningY = []
    testY = []
    for i in range(10):
        if i != test:
            for j in range(int(i * Y_size[0] / 10), int((i + 1) * Y_size[0] / 10)):
                triningY.append(Y[j])

    for j in range(int(test * Y_size[0] / 10), int((test + 1) * Y_size[0] / 10)):
        testY.append(Y[j])

    triningX = np.array(triningX)
    testX = np.array(testX)
    triningY = np.array(triningY)
    testY = np.array(testY)

    return triningX, triningY, testX, testY


def MinMaxNorm(data):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    lines = len(data)
    cols = len(data[0])

    for i in range(lines):
        for j in range(cols):
            if ((data_max[j] - data_min[j]) != 0):
                data[i][j] = (data[i][j] - data_min[j]) / (data_max[j] - data_min[j])


def zscore_train(data):

    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    lines = len(data)
    cols = len(data[0])

    for i in range(lines):
        for j in range(cols):
            if data_std[j] != 0:
                data[i][j] = (data[i][j] - data_mean[j]) / (data_std[j])
    return data, data_mean, data_std


def zscore_test(data, data_mean, data_std):
    lines = len(data)
    cols = len(data[0])

    for i in range(lines):
        for j in range(cols):
            if data_std[j] != 0:
                data[i][j] = (data[i][j] - data_mean[j]) / (data_std[j])
    return data


def encoder(X):
    # define universe of possible input values

    arrayGender = []
    ganderCol = -1

    for i in range(len(X[1])):

        if type(X[0][i]) == str:
            ganderCol = i
            break

    if ganderCol == -1:
        return X

    for i in range(len(X)):
        if X[i][ganderCol] in arrayGender:
            continue
        arrayGender.append(X[i][ganderCol])

    arrayForReplaceGander = []
    new = []
    for i in range(len(X)):
        for j in range(len(arrayGender)):
            new.append(0)
        for j in range(ganderCol):
            new.append(X[i][j])
        for j in range(ganderCol + 1, len(X[1])):
            new.append(X[i][j])
        new.append(1)
        arrayForReplaceGander.append(new)
        new = []

    for i in range(len(X)):
        arrayForReplaceGander[i][arrayGender.index(X[i][ganderCol])] = 1

    return arrayForReplaceGander


def perc_svm_update(x_train, y_train, eta, c, num_iter):
    w = np.zeros((3, x_train.shape[1]))
    for i in range(num_iter):
        rand = np.arange(len(y_train))
        np.random.seed(9001)
        np.random.shuffle(rand)
        x_train = x_train[rand]
        y_train = y_train[rand]
        for x, y in zip(x_train, y_train):
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[int(y), :] = (1 - eta * c) * w[int(y), :] + eta * x
                w[y_hat, :] = (1 - eta * c) * w[y_hat, :] - eta * x
                j = 3 - int(y) - y_hat
                w[j, :] = (1 - eta * c) * w[j, :]
            else:
                for k in range(3):
                    if k != y_hat:
                        w[k, :] = (1 - eta * c) * w[k, :]
    return w


def get_tau(x, y, y_hat, w):
    product1 = np.dot(w[int(y)], x)
    product2 = np.dot(w[y_hat], x)
    product3 = product1 + product2
    loss = max(0.0, 1 - product3)
    if((np.power((np.linalg.norm(x, ord=2)), 2) * 2)!=0):
        return loss / (np.power((np.linalg.norm(x, ord=2)), 2) * 2)
    else:
        return loss / 0.0000001

def pa_update(x_train, y_train, num_iter):
    w = np.zeros((3, x_train.shape[1]))
    for i in range(num_iter):
        rand = np.arange(len(y_train))
        np.random.seed(9001)
        np.random.shuffle(rand)
        x_train = x_train[rand]
        y_train = y_train[rand]
        for x, y in zip(x_train, y_train):
            y_hat = np.argmax(np.dot(w, x))
            tau = get_tau(x, y, y_hat, w)
            if y_hat != y:
                w[int(y), :] = w[int(y), :] + tau * x
                w[y_hat, :] = w[y_hat, :] - tau * x
    return w


def test(y_pred, y_test):
    count = 0
    for y_hat, y in zip(y_pred, y_test):
        if y_hat == y:
            count = count + 1
    return count


def learning(X, Y, eta, c, num_iter):
    success = []
    for i in range(10):
        # arrange the data
        x_train, y_train, x_test, y_test = split_data(X, Y, i)

        # teach the model
        if eta == 0:
            w = pa_update(x_train, y_train, num_iter)
        else:
            w = perc_svm_update(x_train, y_train, eta, c, num_iter)

        # check the model
        count = test(x_test, y_test, w)
        perc = count / y_test.shape[0]
        success.append(perc)

    pred_perc = np.mean(success)
    pred_var = np.var(success)

    return pred_perc, pred_var


def predict(w, x_test):
    y_pred = []
    for x in x_test:
        y_pred.append(np.argmax(np.dot(w, x)))
    return y_pred


def get_perceptron_variables():
    return 0.5, 0, 5


def get_svm_variables():
    return 0.022, 0.0043, 5


def get_pa_variable():
    return 3

def X_from_str_to_flu(X):
    new_X = []
    for i in range(X.shape[0]):
        new = []
        for j in range(X.shape[1]):
            if j==0:
               new.append(str(X[i][j]))
            else:
                new.append(float(X[i][j]))
        new_X.append(new)
    return new_X

def main(argv):


    X  = np.genfromtxt(argv[0], dtype='str', delimiter=",")
    Y_str = np.genfromtxt(argv[1], dtype='str', delimiter=",")
    X_test = np.genfromtxt(argv[2], dtype='str', delimiter=",")
    Y_test = np.genfromtxt("results_file_fixed.txt", dtype='str', delimiter=",")
    X= X_from_str_to_flu(X)
    X_test=X_from_str_to_flu(X_test)

    Y = np.asfarray(Y_str, float)
    Y_test = np.asfarray(Y_test, float)
    # prepare the data for learning (e.i. using one-hot encoder)
    X = preperation(X)
    X_test = preperation(X_test)
    Y = np.array(Y)

    X, data_mean, data_std = zscore_train(X)
    X_test = zscore_test(X_test, data_mean, data_std)

    # calculating w vectors for the models
    perc_eta, perc_tau, perc_num_of_iters = get_perceptron_variables()
    W_perc = perc_svm_update(X, Y, perc_eta, perc_tau, perc_num_of_iters)

    svm_eta, svm_tau, svm_num_of_iters = get_svm_variables()
    W_svm = perc_svm_update(X, Y, svm_eta, svm_tau, svm_num_of_iters)

    W_pa = pa_update(X, Y, get_pa_variable())

    # predict y labels
    perceptron_predictions = predict(W_perc, X_test)
    svm_predictions = predict(W_svm, X_test)
    pa_predictions = predict(W_pa, X_test)

    print( test(Y_test,perceptron_predictions)/len(Y_test), test(Y_test,svm_predictions)/len(Y_test), test(Y_test,pa_predictions)/len(Y_test))
    for i in range(X_test.shape[0]):
        print('perceptron: {}, '.format(perceptron_predictions[i]), 'svm: {}, '.format(svm_predictions[i]), 'pa:',
              pa_predictions[i])



if __name__ == '__main__':
    main(sys.argv[1:])
