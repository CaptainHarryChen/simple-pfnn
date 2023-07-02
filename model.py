import numpy as np

class ELU:
    def __init__(self):
        self.x = None
    
    def __call__(self, x):
        self.x = x
        return np.maximum(x, 0) + np.exp(np.minimum(x, 0)) - 1
    
    def backward(self, grad_y):
        grad_x = np.zeros_like(self.x)
        grad_x[self.x >= 0] = 1
        grad_x[self.x < 0] = np.exp(self.x[self.x < 0])
        grad_x = grad_x * grad_y
        return grad_x


class Linear:
    def __init__(self, input_dim, output_dim):
        k = 1.0 /np.sqrt(input_dim)
        self.weight = np.random.normal(0, k, (4, input_dim, output_dim))
        self.grad_weight = np.zeros_like(self.weight)
        self.bais = np.random.normal(0, k, (4, output_dim,))
        self.grad_bais = np.zeros_like(self.bais)

        self.a = None
        self.x = None
    
    def __call__(self, x, phase):
        self.x = x

        w = (phase * 4) % 1.0
        pindex_1 = (phase * 4).astype(int) % 4
        pindex_0 = (pindex_1-1) % 4
        pindex_2 = (pindex_1+1) % 4
        pindex_3 = (pindex_1+2) % 4

        self.a = np.zeros((x.shape[0], 4), dtype=np.float32)
        self.a[np.arange(x.shape[0]), pindex_0] = -0.5 * w ** 3 + w ** 2 - 0.5 * w
        self.a[np.arange(x.shape[0]), pindex_1] = 1.5 * w ** 3 - 2.5 * w ** 2 + 1
        self.a[np.arange(x.shape[0]), pindex_2] = -1.5 * w ** 3 + 2 * w ** 2 + 0.5 * w
        self.a[np.arange(x.shape[0]), pindex_3] = 0.5 * w ** 3 - 0.5 * w ** 2

        y = np.einsum("ki,lij->klj", x, self.weight) + self.bais
        y = np.sum(self.a.reshape(-1, 4, 1) * y, axis=1)
        return y
    
    def backward(self, grad_y):
        a = self.a[:, :, np.newaxis, np.newaxis]
        x = self.x[:, np.newaxis, :, np.newaxis]
        _grad_y = grad_y[:, np.newaxis, np.newaxis, :]
        self.grad_weight += np.sum(a * x * _grad_y, axis=0)
        a = self.a[:, :, np.newaxis]
        _grad_y = grad_y[:, np.newaxis, :]
        self.grad_bais += np.sum(a * _grad_y, axis=0)

        _weight = np.einsum("ij,jkl->ikl", self.a, self.weight)
        grad_x = np.sum(grad_y.reshape(grad_y.shape[0], 1, -1) * _weight, axis=-1)
        return grad_x
    
    def zero_grad(self):
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bais = np.zeros_like(self.bais)

class SimplePFNN:
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.ELU1 = ELU()
        self.linear2 = Linear(hidden_dim, hidden_dim)
        self.ELU2 = ELU()
        self.linear3 = Linear(hidden_dim, output_dim)

    def __call__(self, x, phase):
        x = self.ELU1(self.linear1(x, phase))
        x = self.ELU2(self.linear2(x, phase))
        x = self.linear3(x, phase)
        return x
    
    def backward(self, grad_y):
        grad_x = self.linear3.backward(grad_y)
        grad_x = self.ELU2.backward(grad_x)
        grad_x = self.linear2.backward(grad_x)
        grad_x = self.ELU1.backward(grad_x)
        grad_x = self.linear1.backward(grad_x)
        return grad_x
    
    def zero_grad(self):
        self.linear1.zero_grad()
        self.linear2.zero_grad()
        self.linear3.zero_grad()
    
    def save(self, filename):
        np.savez(filename, 
            linear1_weight=self.linear1.weight,
            linear1_bais=self.linear1.bais,
            linear2_weight=self.linear2.weight,
            linear2_bais=self.linear2.bais,
            linear3_weight=self.linear3.weight,
            linear3_bais=self.linear3.bais
        )
    
    def load(self, filename):
        with np.load(filename) as f:
            self.linear1.weight = f["linear1_weight"]
            self.linear1.bais = f["linear1_bais"]
            self.linear2.weight = f["linear2_weight"]
            self.linear2.bais = f["linear2_bais"]
            self.linear3.weight = f["linear3_weight"]
            self.linear3.bais = f["linear3_bais"]


if __name__ == "__main__":
    input_dim = 116
    output_dim = 70

    model = SimplePFNN(input_dim, output_dim)