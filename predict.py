# --------------------------------------------------------------------------
# ---  Systems analysis and decision support methods in Computer Science ---
# --------------------------------------------------------------------------
#  Assignment 4: The Final Assignment
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2019
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np

import datetime

def hamming_distance(X, X_train):

    x = np.array(X)
    x_trainT = np.array(X_train)
    xxx = []
    for z in X:
        a = []
        for zz in X_train:
            a.append(sum(abs(z-zz)))
        xxx.append(a)
    return xxx


def sort_train_labels_knn(Dist, y):

    Dist = np.array(Dist)
    order = Dist.argsort(kind='mergesort')
    return y[order]
    # return np.array([y[index_array[int(x / N2), x % N2]] for x in range(N1 * N2)]).reshape((N1, N2))

def p_y_x_knn(y, k):
    def getProbab(y_, neighSet, k_):
        total = 0
        neighSetTemp = neighSet[:k_]
        for neigh in neighSetTemp:
            total += 1 if neigh == y_ else 0
        return total / k_
    ySet = set()
    for x in y:
        for xx in x:
            ySet.add(xx)
    output = []
    for row in y:
        temp = []
        for yi in ySet:
            temp.append(getProbab(yi, row, k))
        output.append(np.array(temp))
    return output

def classification_error(p_y_x, y_true):

    N1 = np.shape(p_y_x)[0]
    result = 0
    for i in range(N1):
        a = p_y_x[i].tolist()
        if (3 - a[::-1].index(max(a)) != y_true[i]):
            result += 1
    return result / N1

def classification(x,x_train,y_train, k):
    sorted_labels = 	sort_train_labels_knn(hamming_distance(x,x_train), y_train)
    pyx = p_y_x_knn(sorted_labels, k)
    output = []
    for p in pyx:
        output.append(p.argmax())
    return output


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    sorted_labels = sort_train_labels_knn(
        hamming_distance(Xval, Xtrain), ytrain)
    errors = list(map(lambda k: classification_error(
        p_y_x_knn(sorted_labels, k), yval), k_values))
    min_index = np.argmin(errors)
    return min(errors), k_values[min_index], errors




def predict(x):
    
    x_train,y_train = pkl.load(open('ula.pkl',mode = 'rb'))
    return np.transpose(np.array(classification(x,x_train,y_train,5)))
