import pandas as pd
from sklearn.svm import SVC


def lin(df):
    # Split the data into training and testing sets
    X = df[['x1', 'x2', 'x3']]
    y = df['y']

    # Train the classifier on the training data

    clf = SVC(kernel='linear')
    clf.fit(X, y)

    # Predict the labels of the testing data
    y_pred = clf.predict(X)

    # Calculate the classification accuracy
    accuracy = sum(y_pred == y) / len(y)
    return accuracy


def getAccuracy(df, a, b):
    X = df[[a, b, 'd']]
    y = df['y']
    clf = SVC(kernel='linear')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    return sum(y_pred == y) / len(y)


def createData(list):
    d = {'x1': [], 'x2': [], 'x3': [], 'd': [], 'y': []}
    for a in list:
        d['x1'].append(a[0])
        d['x2'].append(a[1])
        d['x3'].append(a[2])
        d['d'].append(1)
        d['y'].append(a[3])
    return pd.DataFrame(d)


def forward(df):
    acc = [0, 0, 0]
    for i, x in enumerate(['x1', 'x2', 'x3']):
        for j in range(0, len(df[x])):
            if df[x][j] == -1:
                if df['y'][j] == -1:
                    acc[i] += 1
                else:
                    acc[i] -= 1
            else:
                if df['y'][j] == 1:
                    acc[i] += 1
                else:
                    acc[i] -= 1
        acc[i] = abs(acc[i])
    d = ['x1', 'x2', 'x3']
    if acc.count(max(acc)) != 1:
        return -1
    return d[acc.index(max(acc))]


def getTheBestTwo(df):
    acc = [0, 0, 0]

    acc[0] = getAccuracy(df, 'x1', 'x2')
    acc[1] = getAccuracy(df, 'x1', 'x3')
    acc[2] = getAccuracy(df, 'x2', 'x3')
    max_index = acc.index(max(acc))
    if acc.count(max(acc)) != 1:
        return -1, -1

    if max_index == 0:
        return 'x1', 'x2'
    elif max_index == 1:
        return 'x1', 'x3'
    else:
        return 'x2', 'x3'


def backward(df, f):
    index1, index2 = getTheBestTwo(df)
    if index1 == -1:
        return -1
    acc = {index1: 0, index2: 0}
    for i in acc:
        for j in range(0, len(df[i])):
            if df[i][j] == -1:
                if df['y'][j] == -1:
                    acc[i] += 1
                else:
                    acc[i] -= 1
            else:
                if df['y'][j] == 1:
                    acc[i] += 1
                else:
                    acc[i] -= 1
        acc[i] = abs(acc[i])

    if acc[index1] == acc[index2]:
        if f != index1 and f != index2:
            return index1
        return -1

    return max(acc, key=acc.get)


def chackAcc(l):
    df = createData(l)
    if lin(df) == 1:
        f = forward(df)
        if f != -1:
            b = backward(df, f)
            if b != -1 and f != b:
                print("forward: ", f, "backward: ", b)
                print(df)


def allPosibel(num):
    l = []
    v = {-1, 1}
    for a1 in v:
        for a2 in v:
            for a3 in v:
                l.append([a1, a2, a3, num])
    return l


a = allPosibel(1)
b = allPosibel(1)
c = allPosibel(1)
d = allPosibel(-1)
e = allPosibel(-1)
for ai in a:
    print(ai)
    for bi in b:
        for ci in c:
            for di in d:
                l = [ai, bi, ci, di]
                chackAcc(l)
                for ei in e:
                    l = [ai, bi, ci, di, ei]
                    chackAcc(l)
