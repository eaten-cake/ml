import numpy as np

class LinearRegression:

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def forward(self, X):
        return np.dot(X, self.weights.T) + self.bias
    
    def fit(self, X : np.array, y : np.array):
        n_samples = X.shape[0]
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0

        # 梯度下降
        for epoch in range(self.epochs):
            y_pred = self.forward(X)

            loss = (1 / n_samples) * np.sum((y_pred - y) ** 2)
            if (epoch + 1) % 100 == 0:
                print("Epoch %d Loss: %.4f" % (epoch + 1, loss))
            
            dw = (2 / n_samples) * np.dot(X.T, y_pred - y)
            
            db = (2 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(100, 1)
    # y = 2x + 4
    y = 2 * X + 4 + np.random.randn(100, 1) * 0.1
    lr = LinearRegression(learning_rate=0.01, epochs=1000)
    lr.fit(X, y)
    print("Weights: %.4f, Bias: %.4f" % (lr.weights, lr.bias))
    print("y = %.4fx + %.4f" % (lr.weights, lr.bias))
        
