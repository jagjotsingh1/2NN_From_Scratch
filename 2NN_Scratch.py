# load libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Class for creating the 2 layer Neural Net
class FullyConnectedNet():
    def __init__(self, hidden_dim, input_dim=3072, num_classes=10):
        super(FullyConnectedNet, self).__init__()
        self.params = {}
        self.params['W1'] = np.random.randn(input_dim, hidden_dim)/np.sqrt(input_dim/2)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes)/np.sqrt(hidden_dim/2)
        self.params['b2'] = np.zeros(num_classes)
        self.grads = {}
        self.grads['W1'] = np.zeros_like(self.params['W1'])
        self.grads['b1'] = np.zeros_like(self.params['b1'])
        self.grads['W2'] = np.zeros_like(self.params['W2'])
        self.grads['b2'] = np.zeros_like(self.params['b2'])

    # Relu Activation function
    def relu(self,Z):
        return np.maximum(0,Z)

    # Relu Derivative used for backpropogation
    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    # Forward Pass function
    def forward(self, x):
        """Forward pass: linear -- ReLU -- linear
        """
        # Neurons pass forward
        Z1 = x.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']

        # Storing data within cache, and scores aka z2 output
        cache = (x, Z1, A1, Z2)
        scores = Z2

        return scores, cache

    # Backpropogation using the dscores calculated from cross_entropy_loss
    def backward(self, dscores, cache):
        """Backward pass: update self.grads['W1'], self.grads['b1'],
        self.grads['W2'], self.grads['b2']
        """
        # Working backwards from neurons
        dz2 = dscores
        dA1 = dz2.dot(self.params['W2'].T)
        dz1 = dA1 * self.dRelu(cache[1])
        dA0 = dz1.dot(self.params['W1'].T)

        # Updating internal parameters
        self.grads['W2'] = cache[1].T.dot(dz2)
        self.grads['W1'] = cache[0].T.dot(dz1)
        self.grads['b1'] = dz1.sum(keepdims= False)
        self.grads['b2'] = dz2.sum(keepdims= False)

        return
#-----------------------------------------------------------------------------#
# Functions needed for back propogation
def cross_entropy_loss(scores, y, eps=1e-8):
          scores -= scores.max()
          exp = np.exp(scores)
          exp = np.maximum(exp, eps)
          logits = - scores[range(scores.shape[0]), y] + np.log(exp.sum(axis=1))
          loss = logits.mean()
          dscores = np.zeros_like(scores)
          dscores[range(scores.shape[0]), y] = -1
          dscores += exp/exp.sum(axis=1, keepdims=True)
          dscores /= scores.shape[0]
          return loss, dscores

def gradient_descent(params, grads, lr):
          params["W1"] = params["W1"] - lr * grads["W1"]
          params["b1"] = params["b1"] - lr * grads["b1"]
          params["W2"] = params["W2"] - lr * grads["W2"]
          params["b2"] = params["b2"] - lr * grads["b2"]
          return
#-----------------------------------------------------------------------------#
# load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# preprocess the data
x_train = x_train.reshape(x_train.shape[0], -1).astype('float')
x_test = x_test.reshape(x_test.shape[0], -1).astype('float')
y_train = y_train.squeeze()
y_test = y_test.squeeze()

x_train = x_train - x_train.mean(axis=1, keepdims=True)
x_train = x_train / x_train.std(axis=1, keepdims=True)
x_test = x_test - x_test.mean(axis=1, keepdims=True)
x_test = x_test / x_test.std(axis=1, keepdims=True)
#--------------------------Testing Implementation-----------------------------#
# Creating model object and creating lists to store acc, loss, and acc_val
hidden_dim = 200
model = FullyConnectedNet(hidden_dim=hidden_dim)
loss_history = []
acc_train_history = []
acc_val_history = []

# Loading initial test data and predictions
y_pred, _ = model.forward(x_train)
acc_train = np.mean(y_pred.argmax(axis=1) == y_train)
y_pred, _ = model.forward(x_test)
acc_test = np.mean(y_pred.argmax(axis=1) == y_test)
print(f'Before training, training accuracy={acc_train}, test accuracy={acc_test}')

#----------------------------Training the Model--------------------------------#
num_iters = 2000
batch_size = 500
lr = 1e-2
print_every = num_iters//20
for i in range(num_iters):
    idx = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[idx]
    y_batch = y_train[idx]
    scores, cache = model.forward(x_batch)
    loss, dscores = cross_entropy_loss(scores, y_batch)
    model.backward(dscores, cache)
    gradient_descent(model.params, model.grads, lr=lr)
    loss_history.append(loss.item())
    acc_train = np.mean(scores.argmax(axis=1) == y_batch)
    acc_train_history.append(acc_train.item())
    # test accuracy
    idx = np.random.choice(x_test.shape[0], batch_size)
    x_batch = x_test[idx]
    y_batch = y_test[idx]
    scores, cache = model.forward(x_batch)
    acc_val = np.mean(scores.argmax(axis=1) == y_batch)
    acc_val_history.append(acc_val.item())
    if i == 0 or i == num_iters-1 or (i+1)%print_every == 0:
        print(f'{i+1} loss={loss}, acc_train={acc_train}, acc_val={acc_val}')
#-------------------------------Plotting Data---------------------------------#
plt.plot(loss_history)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
plt.plot(acc_train_history, 'b-', label='train')
plt.plot(acc_val_history, 'g-', label='val')
plt.xlabel('iteration')
plt.ylabel('acc')
plt.legend()
plt.show()
#------------------------------------------------------------------------------#
# Predictions after training
y_pred, _ = model.forward(x_train)
acc_train = np.mean(y_pred.argmax(axis=1) == y_train)
y_pred, _ = model.forward(x_test)
acc_test = np.mean(y_pred.argmax(axis=1) == y_test)
print(f'After training, training accuracy={acc_train}, test accuracy={acc_test}')
