# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:28:58 2022

@author: Lenovo
"""
import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self,X, y, W, b):
        self.X = X
        self.y = y
        self.W = W
        self.b = b
        print("Initial value: W =",W,"b : =",b)

        
    
    def model(self, X, W, b):
        ypred =  np.matmul(self.X, self.W) + self.b
        print("ypred =",ypred)
        print("y =",y)
        loss = np.sum((self.y.reshape(ypred.shape) - ypred) ** 2)/len(self.y)
        rmse = np.sqrt(loss)
        #print("Initial y - prediction: ",ypred)
        print("Initial Loss: ",rmse)
    
    def train(self, epoch, alpha):
        for i in range(epoch):
            self.W, self.b, self.final_loss = gradient(self.X, self.y, self.W, self.b, alpha)
            #self.W, self.b, self.final_loss = stochastic(self.X, self.y, self.W, self.b, alpha)

        print(self.final_loss)
        
    
def gradient(X, y, W, b, alpha):
    ypred =  np.matmul(X, W) + b
    print(ypred)

    grad_w = sum((y - ypred.reshape(y.shape))*X)
    grad_b = sum(y - ypred.reshape(y.shape))
        
    W = W + alpha * grad_w
    b = b + alpha * grad_b
        
    ypred = np.matmul(X, W) + b
        
    loss = np.sum((y - ypred.reshape(y.shape)) ** 2)/(len(y))
    rmse = np.sqrt(loss)
    print(rmse)
        
    return (W, b, rmse)
    
def stochastic(X, y, W, b, alpha):
    ypred =  np.matmul(X, W) + b
    
    grad_W = (y - ypred.reshape(y.shape))*X
    grad_b = (y - ypred.reshape(y.shape))
    
    
    for i in range(len(X)):
        # Set update rule
        W = W - alpha * grad_W[i]
        b = b - alpha * grad_b[i]
        
    loss = np.sum((y - ypred.reshape(y.shape)) ** 2)/(len(y))
    rmse = np.sqrt(loss)
    print(rmse)
        
    return (W, b, rmse)

def synthetic_data(rows, cols):
    X = np.random.normal(0,1,(rows,cols))
    
    row, col = X.shape
    
    W = np.reshape(np.random.rand(col),(col,1))
    b = np.zeros(col)
    y = np.matmul(X, W) + b
    y += np.random.normal(0, 0.01, y.shape)
        
    return X, y.reshape(-1, 1), W , b

def read_data():
    path = "data/data.csv"
    df = pd.read_csv(path)
    X = df["X"][:, np.newaxis]
    y = df["y"][:, np.newaxis]
    
    W = np.random.rand(1)
    b = np.zeros(1)
    
    return X, y, W, b
        
#X, y ,W , b = synthetic_data()
if __name__ == "__main__":
    #X, y , W, b = random_data()
    X, y , W, b = read_data()
    lr = LinearRegression(X, y , W, b)
    lr.model(X, W, b)
    lr.train(20, 0.001)

