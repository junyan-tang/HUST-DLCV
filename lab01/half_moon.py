from sklearn.datasets import make_moons
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np

# set the number of samples which would be generated
n_samples = 12000
train_num = 10000

# generate 2d classification dataset
X, y = make_moons(n_samples, noise=0.1, shuffle=True)

# scatter plot,dots colored by class value
# df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
# colors = {0: 'green', 1: 'yellow'}
# fig, ax = pyplot.subplots()
# grouped = df.groupby('label')
# for key, group in grouped:
#     group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
# pyplot.figure()
# pyplot.show()

# divide the test and training set
train_X, train_y = np.array(X[:train_num]), np.array(y[:train_num])
test_X, test_y = np.array(X[train_num:]), np.array(y[train_num:])

# define some activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def ReLU(x):
    return np.maximum(0, x)
def TanH(x):
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))
def LeakyReLU(x):
    return np.maximum(0.1 * x, x)

# define some backward dx functions
def sig_back(x):
    value = sigmoid(x)
    return (1.0 - value) * value
def TanH_back(x):
    value = TanH(x)
    return 1 - np.power(value, 2)
def ReLU_back(x):
    return np.where(x < 0, 0, 1)
def Leaky_back(x):
    return np.where(x < 0, 0.1, 1)

# define the neural network
class BP_NN:
    """
    A neural network with:
      - 2 inputs
      - a hidden layer with several neurons
      - an output layer with 2 outputs
    """

    def __init__(self, learning_rate, inputs_dim, num_H, num_O, is_training=True):
        self.learning_rate = learning_rate
        self.num_H = num_H
        self.num_O = num_O
        self.training = is_training
        self.w1 = np.random.randn(inputs_dim, self.num_H) * 0.01
        self.w2 = np.random.randn(self.num_H, self.num_O) * 0.01
        self.b1 = np.zeros((1, self.num_H))
        self.b2 = np.zeros((1, self.num_O))

    def feedforward(self, x, y):
        example_num = x.shape[0]
        y_ext = np.zeros((example_num, self.num_O))
        for i, j in enumerate(y, 0):
            y_ext[i, j] = 1
        z1 = np.dot(x, self.w1) + self.b1
        a1 = TanH(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        loss = -(y_ext * np.log(probs)).sum() / example_num
        return z1, a1, z2, probs, example_num

    def backward(self, x, y):
        z1, a1, z2, probs, example_num = self.feedforward(x, y)
        delta3 = probs
        delta3[range(example_num), y] -= 1
        dw2 = np.dot(a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.w2.T) * TanH_back(a1)
        dw1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # gradient descent and weights are updated
        self.w1 -= self.learning_rate * dw1
        self.w2 -= self.learning_rate * dw2
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2

    # predict the results of testing set
    def predict(self, x):
        z1, a1, z2, probs, _ = self.feedforward(x, y)
        return np.argmax(probs, axis=1)

    # calculate the accuracy of the prediction
    def accuracy(self, x, y_input):
        y_pred = self.predict(x)
        # 2000 is the number of test samples
        y_input = y_input.reshape(2000, 1)
        return 1 - np.sum(abs(y_pred - y_input)) / len(y_input)


net = BP_NN(0.001, 2, 5, 2)

# mini-batch
epoch = 1000
batch_size = 20
for i in range(epoch):
    for j in range(0, train_num, batch_size):
        net.backward(train_X[j:j + batch_size], train_y[j:j + batch_size])

# calculate the accuracy on test set
# accuracy = net.accuracy(test_X, test_y)
# print(accuracy)

# visualize classification results
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = net.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# it takes nearly one minute to see the result
pyplot.contourf(xx, yy, Z, cmap="Greens")
pyplot.scatter(X[:, 0], X[:, 1], c=y, cmap="summer")
pyplot.show()
