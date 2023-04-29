import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    return np.maximum(x, 0)

def gradRelu(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

def compute(x, W, b):
    return np.matmul(x, W) + b

def averageCE(target, prediction):
    return -1*np.mean(target*np.log(prediction))

def gradCE(target, prediction):
    return prediction-target

def dL_dW_o(target, prediction, h):
    return np.matmul(np.transpose(h), gradCE(target, prediction))

def dL_db_o(target, prediction):
    return np.matmul(np.ones((1, target.shape[0])), gradCE(target, prediction))

def dL_dW_h(target, prediction, W_o, x, W_h, b_h):
    hidden_in = compute(x, W_h, b_h)
    d_r = gradRelu(hidden_in)
    reLU = relu(compute(x, W_h, b_h))
    return np.matmul(np.transpose(x), d_r*np.matmul(gradCE(target, prediction), np.transpose(W_o)))

def dL_db_h(target, prediction, W_o, x, W_h, b_h):
    hidden_in = compute(x, W_h, b_h)
    gradRelu(hidden_in)
    d_r = relu(hidden_in)
    return np.matmul(np.ones((1, hidden_in.shape[0])), d_r*np.matmul(gradCE(target, prediction), np.transpose(W_o)))

def accuracy_cal(pred,y):
    prediction  = pred.argmax(axis = 1) 
    target = y.argmax(axis=1)
    return np.mean(np.equal(prediction, target)==True)


#LEARNING
train_loss = []
valid_loss = []
test_loss = []

train_accuracy = []
valid_accuracy = []
test_accuracy = []

#load data
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
validData = validData.reshape((validData.shape[0],validData.shape[1]*validData.shape[2])) 
testData = testData.reshape((testData.shape[0],testData.shape[1]*testData.shape[2]))
newTrain, newValid, newTest = convertOneHot(trainTarget, validTarget, testTarget)

#constants
epochs = 200
H = 1000
gamma = 0.9
alpha = 0.0000001

#bias vectors
b_h = np.zeros((1,H))
b_h_updated = b_h
b_o = np.zeros((1,10))
b_o_updated = b_o

#weight matrices
#W_h = np.random.rand((np.shape(trainData)[0],np.shape(trainTarget)[0]))*np.sqrt(1/(H+784))
#W_o = np.random.rand((np.shape(trainData)[0],np.shape(trainTarget)[0]))*np.sqrt(1/(H+10))
W_h = np.random.normal(0, np.sqrt(1/(H+trainData.shape[0])), (trainData.shape[1],H))
W_o = np.random.normal(0, np.sqrt(1/(H+10)), (H,10))

#momentum matrices
v_h =  np.full(np.shape(W_h), 1e-5)
v_o = np.full(np.shape(W_o), 1e-5)

#training loop
for i in range(epochs):
    print(i)
    x_relu_input = compute(trainData, W_h, b_h)
    x_hidden = relu(x_relu_input)
    x_softmax_input = compute(x_hidden, W_o, b_o)
    x_p = softmax(x_softmax_input)
    x_loss = averageCE(newTrain, x_p)
    train_loss.append(x_loss)
    x_accuracy = accuracy_cal(x_p, newTrain)
    train_accuracy.append(x_accuracy)

    v_relu_input = compute(validData, W_h, b_h)
    v_hidden = relu(v_relu_input)
    v_softmax_input = compute(v_hidden, W_o, b_o)
    v_p = softmax(v_softmax_input)
    v_loss = averageCE(newValid, v_p)
    valid_loss.append(v_loss)
    v_accuracy = accuracy_cal(v_p, newValid)
    valid_accuracy.append(v_accuracy)

    t_relu_input = compute(testData, W_h, b_h)
    t_hidden = relu(t_relu_input)
    t_softmax_input = compute(t_hidden, W_o, b_o)
    t_p = softmax(t_softmax_input)
    t_loss = averageCE(newTest, t_p)
    test_loss.append(t_loss)
    t_accuracy = accuracy_cal(t_p, newTest)
    test_accuracy.append(t_accuracy)

    #back propagation
    v_o = gamma*v_o + alpha*dL_dW_o(newTrain, x_p, x_hidden)
    W_o -= v_o
    b_o_updated = gamma*b_o_updated + alpha*dL_db_o(newTrain, x_p)
    b_o -= b_o_updated

    v_h = gamma*v_h + alpha*dL_dW_h(newTrain, x_p, W_o, trainData, W_h, b_h)
    W_h -= v_h
    b_h_updated = gamma*b_h_updated + alpha*dL_db_h(newTrain, x_p, W_o, trainData, W_h, b_h)
    b_h -= b_h_updated


print(train_loss[199])
print(valid_loss[199])
print(test_loss[199])
print(train_accuracy[199])
print(valid_accuracy[199])
print(test_accuracy[199])

#plot
x_axis = range(200)
plt.plot(x_axis, train_loss, 'r', x_axis, valid_loss, 'b')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(["Training Loss","Valid Loss"],loc='upper right')
plt.title("Data Losses with gamma=0.99, H=1000")
plt.show()
plt.plot(x_axis, train_accuracy, 'r', x_axis, valid_accuracy, 'b')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(["Training Accuracy","Valid Accuracy"], loc='lower right')
plt.title("Data Accuracies with gamma=0.99, H=1000")
plt.show()

print('dont know where im in 5 but im young and alive')
