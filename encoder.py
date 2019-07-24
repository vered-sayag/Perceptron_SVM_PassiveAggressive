# -- coding: utf-8 --
"""
Created on Fri May 10 10:52:34 2019

@author: דוד רגב
"""

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import argmax

def MinMaxNorm(data):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    lines = len(data)
    cols = len(data[0])

    for i in range(lines):
        for j in range(cols):
            if((data_max[j] - data_min[j])!=0):
                data[i][j] = (data[i][j] - data_min[j]) / (data_max[j] - data_min[j])

def zscor(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)

    lines = len(data)
    cols = len(data[0])

    for i in range(lines):
        for j in range(cols):
            if ((data_std[j]) != 0):
                data[i][j] = (data[i][j] - data_mean[j]) / (data_std[j])
    return data
def perc_learning(x_train, y_train, eta, num_iter):
    w = np.zeros((3, x_train.shape[1]))  # TODO - how to get 3 generically
    for i in range(num_iter):
        #indexes = np.random.permutation(x_train.shape[0])
        #x_train, y_train = x_train[indexes], y_train[indexes]
        for x, y in zip(x_train, y_train):
            # for i in range(x_train.shape[0]):
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[int(y), :] = w[int(y), :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
    return w


def perc_test(x_test, y_test, w):
    count = 0
    for x, y in zip(x_test, y_test):
        y_pred = np.argmax(np.dot(w, x))
        if y_pred == y:
            count = count + 1
    return count


def perceptron(X, Y, eta, num_iter):
    success = []
    for i in range(10):
        # arrange the data
        x_train, y_train, x_test, y_test = split_data(X, Y, i)

        # teach the model
        w = perc_learning(x_train, y_train, eta, num_iter)
        #        print("iter {}".format(i), w)
        # check the model
        count = perc_test(x_test, y_test, w)
        perc = count / y_test.shape[0]
        success.append(perc)

    pred_perc = np.mean(success)
    pred_var = np.var(success)

    return pred_perc, pred_var


def preperation(X):
    X = encoder(X)
    #X = MinMaxNorm(X)
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


def encoder(X):
    # define universe of possible input values
    X = X.rename_axis('ID')
    X = X.values
    arrayGender = []
    matrix_size = X.shape
    ganderCol = -1

    for i in range(matrix_size[1]):

        if type(X[0][i]) == str:
            ganderCol = i
            break

    if (ganderCol == -1):
        return X

    for i in range(matrix_size[0]):
        if X[i][ganderCol] in arrayGender:
            continue
        arrayGender.append(X[i][ganderCol])

    arrayForReplaceGander = []
    new = []
    for i in range(matrix_size[0]):
        for j in range(len(arrayGender)):
            new.append(0)
        for j in range(ganderCol):
            new.append(X[i][j])
        for j in range(ganderCol + 1, matrix_size[1]):
            new.append(X[i][j])
        new.append(1)
        arrayForReplaceGander.append(new)
        new = []

    for i in range(matrix_size[0]):
        arrayForReplaceGander[i][arrayGender.index(X[i][ganderCol])] = 1

    return arrayForReplaceGander


def plot_grid(X, Y, Z, xlabel, ylabel, zlabel ):
    #Axes3D = Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_trisurf(X, Y, Z, cmap=plt.cm.RdYlGn, linewidth=0)
    fig.colorbar(surf)

    #ax.xaxis.set_major_locator(plt.ticker.MaxNLocator(5))
    #ax.yaxis.set_major_locator(plt.ticker.MaxNLocator(6))
    #ax.zaxis.set_major_locator(plt.ticker.MaxNLocator(5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    fig.tight_layout()

    plt.show()


def main():
    X = pd.read_csv('train_x.txt', sep=",", header=None)
    #    X.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
    #                      'shucked weight', 'Viscera weight', 'shell weight']
    Y = pd.read_csv('train_y.txt', header=None)
    #    Y.columns = ['Label']

    # prepare the data for learning (e.i. using one-hot encoder)
    X = preperation(X)

    array_eta = []
    array_iter = []
    array_pred_perc = []


    array_pred_var = []
    for num_iter in range(1, 20):
        #for i in range(1,10000,100):
        # eta = 0.0001*(10**i)
        eta = 10000000
        pred_perc, pred_var = perceptron(X, Y, eta, num_iter)
        array_eta.append(eta)
        array_iter.append(num_iter)
        array_pred_perc.append(pred_perc * 100)
        array_pred_var.append(pred_var)

    plt.figure()
    plt.plot(array_iter,array_pred_perc)
    plt.show()
    plt.figure()
    plt.plot(array_iter, array_pred_var)
    plt.show()
    #plot_grid(array_eta, array_iter, array_pred_perc, "eta", "epochs", "success rate")
    #plot_grid(array_eta, array_iter, array_pred_var, "eta", "epochs", "variance")


main()