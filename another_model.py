# the number of outputlayer is 1, use sigmoid as softmax
def feedforward(self, x):
    z1 = np.dot(x, self.w1) + self.b1
    a1 = TanH(z1)
    z2 = np.dot(a1, self.w2) + self.b2
    z = sigmoid(z2)
    return z1, a1, z2, z

# the loss function is changed to MSE
def backward(self, x, y):
    z1, a1, z2, z = self.feedforward(x)
    # 20 is batch size
    y = y.reshape(1, 20)
    loss = (1.0 / x.shape[0]) * sum((y.T - z) ** 2)
    probs = z - y.T
    delta3 = probs
    dw2 = np.dot(a1.T, delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = np.dot(delta3, self.w2.T) * TanH_back(a1)
    dw1 = np.dot(x.T, delta2)
    db1 = np.sum(delta2, axis=0)


# calculate the accuracy of the prediction
def accuracy(self, x, y_input):
    y_pred = np.floor(self.predict(x) * 1.9999)
    # 2000 is the number of test samples
    y_input = y_input.reshape(2000, 1)
    return 1 - np.sum(abs(y_pred - y_input)) / len(y_input)
