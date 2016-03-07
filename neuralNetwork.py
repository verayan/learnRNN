__author__ = 'wyan'


import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


# generate a pseudo dataset
np.random.seed(0)
X, y = make_moons(200, noise=0.20)


# use logistic regression to fit
clf = linear_model.LogisticRegressionCV()
clf.fit(X, y)
h = 0.02
X1_min, X1_max = X[:,0].min()-.5, X[:,0].max() + .5
X2_min, X2_max = X[:,1].min() - .5, X[:,1].max() + .5
x1,x2 = np.meshgrid(np.arange(X1_min, X1_max, h), np.arange(X2_min, X2_max, h))
Z = clf.predict(np.c_[x1.ravel(), x2.ravel()])
Z = Z.reshape(x1.shape)

# show the decision boundary
plt.pcolormesh(x1, x2, Z, cmap = plt.cm.Paired)
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()


# implementation of neural network
num_examples = len(X)
input_dim = 2
output_dim = 2
epsilon = 0.01
reg = 0.01

# helpful function to evaluate the total loss
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    return 1./num_examples * data_loss

# helper function to predict an output(0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    return np.argmax(probs, axis = 1)

def build_model(nn_hdim, num_passes = 20000, print_loss = False):
    np.random.seed(0)
    W1 = np.random.randn(input_dim, nn_hdim) / np.sqrt(input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, output_dim))
    model = {}
    for i in range(0, num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
        # back propagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg * W2
        dW1 += reg * W1
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, calculate_loss(model))
    return model






