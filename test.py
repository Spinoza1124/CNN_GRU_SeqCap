import numpy as np

class BatchNormalization:
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
        self.gamma = 1.0
        self.beta = 0.0
        self.running_mean = None
        self.running_var = None
        self.momentum = 0.9
    
    def forward(self, x, training=True):
        batch_size, features = x.shape

        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            x_normalized = (x - mean) / np.sqrt(var + self.epsilon)

            if self.running_mean is None:
                self.running_mean = np.zeros_like(mean)
                self.running_var = np.zeros_like(var)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            if self.running_mean is None or self.running_var is None:
                raise ValueError("Running mean/variance not initialized. Run training forward pass first.")
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        out = self.gamma * x_normalized + self.beta

        return out

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.randn(32, 10)
    print(x)

    bn = BatchNormalization()

    output = bn.forward(x, training=True)
    print("Output shape:", output.shape)
    print("First few normalized values:\n", output[:2, :5])
            
