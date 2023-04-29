import numpy as np
import matplotlib.pyplot as plt
import math

def data():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, trainTarget, validData, validTarget, testData, testTarget

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(W, b, x, y, reg):
    n = np.shape(y)[0]
    z = np.dot(x,W) + b
    lce = -1/n*np.sum(y*np.log(sigmoid(z)) + (1-y)*np.log(1-sigmoid(z)))
    lw = (reg/2)*(np.linalg.norm(W)**2)
    return lce+lw

def grad_loss(W, b, x, y, reg):
    n = np.shape(y)[0]
    z = np.dot(x,W)+b
    gradW = np.dot(np.transpose(x),(sigmoid(z)-y))/n + reg*W
    gradB = np.sum(sigmoid(z)-y)/n
    return gradW, gradB

def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol):
    for i in range(epochs):
        gradW, gradB = grad_loss(W, b, x, y, reg)
        newW = W - alpha*gradW
        newB = b - alpha*gradB
        if np.linalg.norm(W-newW)<error_tol:
            return newW, newB
        else:
            W = newW
            b = newB
    return W,b

def accuracy_cal(W,b,x,y):
    z = np.dot(x,W)+b
    prediction = sigmoid(z)>=0.5
    return np.mean(prediction==y)

def all_grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, validData, validTarget, testData, testTarget):
    n = np.shape(y)[0]

    x_losses = [loss(W, b, x, y, reg)]
    x_accuracy = [accuracy_cal(W,b,x,y)]

    v_losses = [loss(W, b, validData, validTarget, reg)]
    v_accuracy = [accuracy_cal(W,b,validData,validTarget)]

    t_losses = [loss(W, b, testData, testTarget, reg)]
    t_accuracy = [accuracy_cal(W,b,testData,testTarget)]

    for i in range(epochs):
        gradW, gradB = grad_loss(W, b, x, y, reg)
        newW = W - alpha*gradW
        newB = b - alpha*gradB

        x_losses.append(loss(newW, newB, x, y, reg))
        x_accuracy.append(accuracy_cal(newW,newB,x,y))

        v_losses.append(loss(newW, newB, validData, validTarget, reg))
        v_accuracy.append(accuracy_cal(newW,newB,validData,validTarget))

        t_losses.append(loss(newW, newB, testData, testTarget, reg))
        t_accuracy.append(accuracy_cal(newW,newB,testData,testTarget))

        if np.linalg.norm(W-newW)<error_tol:
            return newW, newB, x_losses, x_accuracy, v_losses, v_accuracy, t_losses, t_accuracy
        else:
            W = newW
            b = newB

    print(accuracy_cal(newW,newB,x,y))
    print(accuracy_cal(newW,newB,validData,validTarget))
    print(accuracy_cal(newW,newB,testData,testTarget))

    return W, b, x_losses, x_accuracy, v_losses, v_accuracy, t_losses, t_accuracy

def part1():
    trainData, trainTarget, validData, validTarget, testData, testTarget = data()
    trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
    validData = validData.reshape((validData.shape[0],validData.shape[1]*validData.shape[2])) 
    testData = testData.reshape((testData.shape[0],testData.shape[1]*testData.shape[2]))
    W = np.random.normal(0,0.5,(trainData.shape[1],1))
    b=0
    reg = 0.5    
    alpha = 0.005
    error_tol = 0.0000001
    epochs = 5000
    W, b, x_losses, x_accuracy, v_losses, v_accuracy, t_losses, t_accuracy = all_grad_descent(W,b,trainData,trainTarget,alpha,epochs,reg,error_tol, validData, validTarget, testData, testTarget)
    x_range = range(5001)

    plt.plot(x_range, x_losses, 'r', x_range, v_losses, 'b', x_range, t_losses, 'g')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(["Training Loss","Valid Loss", "Test Loss"],loc='upper right')
    plt.title("Data Losses with ⍺=0.005, ƛ = 0.5")
    plt.show()
    plt.plot(x_range, x_accuracy, 'r', x_range, v_accuracy, 'b', x_range, t_accuracy, 'g')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend(["Training Accuracy","Valid Accuracy", "Test Accuracy"], loc='lower right')
    plt.title("Data Accuracies with ⍺=0.005, ƛ = 0.5")
    plt.show()

part1()
