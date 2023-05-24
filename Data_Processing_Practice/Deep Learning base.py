##Perceptron Gate

import pandas as pd
import matplotlib as plt
import matplotlib.pylab as plt
import numpy as np

def AND(x1,x2) :
    b = -0.7
    if ((x1!=1)&(x1!=0))|((x2!=1)&(x2!=0)) :
        print("invalid input")
    else :
        w = np.array([0.5, 0.5])
        x = np.array([x1,x2])
        std = np.sum(w*x) + b
        if std > 0 :
            return(1)
        else : return(0)

def OR(x1,x2) :
    b = -0.2
    if ((x1!=1)&(x1!=0))|((x2!=1)&(x2!=0)) :
        print("invalid input")
    else :
        w = np.array([0.5, 0.5])
        x = np.array([x1,x2])
        std = np.sum(w*x) + b
        if std > 0 :
            return(1)
        else : return(0)

def NAND(x1,x2) :
    b = -0.7
    if ((x1!=1)&(x1!=0))|((x2!=1)&(x2!=0)) :
        print("invalid input")
    else :
        w = np.array([0.5, 0.5])
        x = np.array([x1,x2])
        std = -(np.sum(w*x) + b)
        if std > 0 :
            return(1)
        else : return(0)

def XOR(x1, x2) :
    a = NAND(x1, x2)
    b = OR(x1,x2)
    c = AND(a,b)
    return c


def step_function(x) :
    return np.array(x > 0 , dtype = np.int)


def sigmoid(x) : 
    return 1/(1+np.exp(-x))


def relu(x) : 
    return np.maximum(0,x)


def st_fr(X) :
    W1 = np.array([[0.1, 0.3, 0.6],
                   [0.4, 0.1, 0.6]])
    B1 = np.array([0.3,0.6,0.1])
    if len(X) != 2 :
        return print("Incorrect data set")
    else : 
        A1 = np.dot(X, W1) + B1
        Y1 = sigmoid(A1)
    W2 = np.array([[0.1, 0.3],
                   [0.4, 0.1],
                   [0.9, 0.2]])
    B2 = np.array([0.3,0.6])
    A2 = np.dot(Y1, W2) + B2
    Y2 = relu(A2)
    W3 = np.array([[0.8, 0.9],
                   [0.3, 0.2]])
    B3 = np.array([0.3,-0.4])
    A3 = np.dot(Y2, W3) + B3
    Y3 = sigmoid(A3)
    return Y2

def soft_max(a) :
    exa = np.exp(a)
    y = exa / sum(exa)
    return y

    