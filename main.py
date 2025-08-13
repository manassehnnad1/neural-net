import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

#read csv file
data = pd.read_csv('train.csv')


#create array from csv using numpy
data = np.array(data)

# Shuffle the data and set row and column + 1 dimension
m, n = data.shape
np.random.shuffle(data)

#eliminate overfeeding by setting aside some chunks and transpose
data_asd = data[0:1000].T
Y_dev = data_asd[0]
X_dev = data_asd[1:n]

#define training data from chunk leftovers
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

#output Y_train to be sure my code is working lol
#print(Y_train)

def initialize():
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return w1,w2,b2,b1

# Define ReLU activation function
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)    

#forward propagation for input layer
def forward_prop(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = w2.dot(A1) + b2
    A2 = ReLU(Z2)
    return A1, A2, Z1, Z2

def compute(Y):
    compute_Y = np.zeros((Y.size, Y.max() + 1))
    compute_Y[np.arange(Y.size), Y] = 1
    compute_Y = compute_Y.T
    return compute_Y

def ReLU_prime(Z):
    return (Z > 0)

#backward propagation
def backward_prop(Z1, A1, A2, Z2, W2, X, Y):
    m = Y.size
    one_hot_Y = compute(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/ m * dZ2.dot(A1.T)
    db2 = 1/ m * np.sum(dZ2, 2)
    dZ1 = W2.T.dot(dZ2) 
    dW1 = 1/ m * dZ2.dot(A1.T)
    db1 = 1/ m * np.sum(dZ2, 2)
    return dW1, dW2, db1, db2

def new_params(W1,  b1, W2, b2, dW1, db1, dW2, dB2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * dB2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, axis=0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

#gradient descent
def gradient_descent(X, Y, num_iterations, alpha):
    W1, W2, b2, b1 = initialize()
    for _ in range(num_iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, dW2, db1, db2 = backward_prop(Z1, A1, A2, Z2, W2, X, Y)
        W1, b1, W2, b2 = new_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if _ % 10 == 0:
            print('iteration: ', _)
            print('Accuracy: ', get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)