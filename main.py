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

#forward propagation for input layer
def forward_prop(w1, b1, w2, b2, X):
    Z1 = w1.dot(X) + b1





