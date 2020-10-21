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
        Z1 = x.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']

        #trail and error
        cache = (x, Z1, A1, Z2)
        scores = Z2

        return scores, cache

    # Backpropogation using the dscores calculated from cross_entropy_loss
    def backward(self, dscores, cache):
        """Backward pass: update self.grads['W1'], self.grads['b1'], self.grads['W2'], self.grads['b2']
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
