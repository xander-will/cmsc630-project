import csv
import imageio 
import json
import matplotlib.pyplot as plt
import numpy as np
import re

from glob import glob
from math import sqrt
from random import shuffle
from statistics import mean, mode

def EuclidDist(x, y):
    area = (float(x[1]) - float(y[1]))**2
    perimeter = (float(x[2]) - float(y[2]))**2
    std_dev = (float(x[3]) - float(y[3]))**2
    cell = (float(x[4]) - float(y[4]))**2
    bkgd = (float(x[5]) - float(y[5]))**2

    return sqrt(area + perimeter + std_dev + cell + bkgd)

def Decision(items, test, train, k):
    test = items[test]
    dists = []
    for i in train:
        dist = EuclidDist(test, items[i])
        dists.append((dist, items[i][-1]))

    list.sort(dists, key=lambda x: x[0])
    nn = [i[1] for i in dists[:k]]
    return mode(nn)

def KNN(items, test, train, k):
    accuracy = 0
    for t in test:
        label = Decision(items, t, train, k)
        if label == items[t][-1]:
            accuracy += 1

    return accuracy / len(test)

def CreateSets(groups, i_test):
    test = groups[i_test]
    train = []

    for i in range(10):
        if i != i_test:
            train.extend(groups[i])

    return test, train

def Model(items, k):
    print("---------\nk = ", k)
    group_size = len(items) // 10
    print("Group size:", group_size)
    ids = list(range(len(items)))
    shuffle(ids)

    groups = []
    for i in range(9):
        start = i * group_size
        groups.append(ids[start:start+group_size])
    groups.append(ids[9*group_size:])

    accuracies = []
    for i in range(10):
        test, train = CreateSets(groups, i)
        accuracies.append(KNN(items, test, train, k))
        print(f"test group: {i}, accuracy: {accuracies[-1]}")

    return mean(accuracies)


if __name__ == "__main__":
    items = []
    with open('dataset.csv') as f:
        flag = True
        for row in csv.reader(f):
            if flag:
                items.append(row)
                flag = False
            else:
                flag = True
    
    accuracies = {}
    for k in [1, 3, 5, 7, 9, 15, 21, 35]:
        accuracies[k] = Model(items, k)

    print("---------")
    for key in accuracies:
        print(f"k={key}: {accuracies[key]}")

