import numpy as np

try:
    import cupy as cp
    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False


def get_xp(arr):
    if _GPU_AVAILABLE and cp is not None:
        return cp.get_array_module(arr)
    return np


def _to_gpu(arr):
    if _GPU_AVAILABLE and cp is not None:
        return cp.asarray(arr)
    return arr


class Dense:
    """
    Fully connected linear layer.
    He et al. 2015 - Delving Deep into Rectifiers (He initialization).
    arXiv:1502.01852. https://arxiv.org/abs/1502.01852
    W ~ N(0, sqrt(2 / n_in)).  Forward: z = xW + b.
    """

    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = (np.random.randn(in_features, out_features) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = None
        self.db = None

    def to_gpu(self):
        self.W = _to_gpu(self.W)
        self.b = _to_gpu(self.b)

    def forward(self, x, training=True):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]

    def __call__(self, x, training=True):
        return self.forward(x, training)


class BatchNormalization:
    """
    Batch Normalization.
    Ioffe & Szegedy 2015 - Batch Normalization: Accelerating Deep Network Training.
    arXiv:1502.03167. https://arxiv.org/abs/1502.03167
    Backward uses the analytically exact, numerically stable formulation from the paper.
    Running stats tracked with momentum=0.9 for inference.
    """

    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        self.dgamma = None
        self.dbeta = None
        self._x_norm = None
        self._var = None
        self._mean = None
        self._x = None
        self._N = None

    def to_gpu(self):
        self.gamma = _to_gpu(self.gamma)
        self.beta = _to_gpu(self.beta)
        self.running_mean = _to_gpu(self.running_mean)
        self.running_var = _to_gpu(self.running_var)

    def forward(self, x, training=True):
        xp = get_xp(x)
        if training:
            self._N = x.shape[0]
            self._mean = x.mean(axis=0)
            self._var = x.var(axis=0)
            self._x = x
            self._x_norm = (x - self._mean) / xp.sqrt(self._var + self.eps)
            self.running_mean = (
                self.momentum * self.running_mean + (1.0 - self.momentum) * self._mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1.0 - self.momentum) * self._var
            )
        else:
            self._x_norm = (x - self.running_mean) / xp.sqrt(self.running_var + self.eps)
        return self.gamma * self._x_norm + self.beta

    def backward(self, grad):
        xp = get_xp(grad)
        N = self._N
        x_mu = self._x - self._mean
        std_inv = 1.0 / xp.sqrt(self._var + self.eps)

        dx_norm = grad * self.gamma
        dvar = (dx_norm * x_mu * (-0.5) * std_inv ** 3).sum(axis=0)
        dmean = (dx_norm * (-std_inv)).sum(axis=0) + dvar * (-2.0 / N) * x_mu.sum(axis=0)
        dx = dx_norm * std_inv + dvar * (2.0 / N) * x_mu + dmean / N

        self.dgamma = (grad * self._x_norm).sum(axis=0)
        self.dbeta = grad.sum(axis=0)
        return dx

    def params(self):
        return [self.gamma, self.beta]

    def grads(self):
        return [self.dgamma, self.dbeta]

    def __call__(self, x, training=True):
        return self.forward(x, training)


class LayerNormalization:
    """
    Layer Normalization.
    Ba et al. 2016 - Layer Normalization.
    arXiv:1607.06450. https://arxiv.org/abs/1607.06450
    Normalizes across feature dimension (axis=-1) rather than batch dimension.
    More stable than BatchNorm for small batch sizes.
    """

    def __init__(self, num_features, eps=1e-5):
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.eps = eps
        self.dgamma = None
        self.dbeta = None

    def to_gpu(self):
        self.gamma = _to_gpu(self.gamma)
        self.beta = _to_gpu(self.beta)

    def forward(self, x, training=True):
        xp = get_xp(x)
        self._x = x
        self._mean = x.mean(axis=-1, keepdims=True)
        self._var = x.var(axis=-1, keepdims=True)
        self._x_norm = (x - self._mean) / xp.sqrt(self._var + self.eps)
        return self.gamma * self._x_norm + self.beta

    def backward(self, grad):
        xp = get_xp(grad)
        N = self._x.shape[-1]
        x_mu = self._x - self._mean
        std_inv = 1.0 / xp.sqrt(self._var + self.eps)

        dx_norm = grad * self.gamma
        dvar = (dx_norm * x_mu * (-0.5) * std_inv ** 3).sum(axis=-1, keepdims=True)
        dmean = (dx_norm * (-std_inv)).sum(axis=-1, keepdims=True) + dvar * (-2.0 / N) * x_mu.sum(axis=-1, keepdims=True)
        dx = dx_norm * std_inv + dvar * (2.0 / N) * x_mu + dmean / N

        self.dgamma = (grad * self._x_norm).sum(axis=0)
        self.dbeta = grad.sum(axis=0)
        return dx

    def params(self):
        return [self.gamma, self.beta]

    def grads(self):
        return [self.dgamma, self.dbeta]

    def __call__(self, x, training=True):
        return self.forward(x, training)


def make_norm(norm_type, num_features):
    if norm_type == "batch":
        return BatchNormalization(num_features)
    elif norm_type == "layer":
        return LayerNormalization(num_features)
    else:
        raise ValueError(f"Unknown normalization '{norm_type}'")


class Dropout:
    """
    Inverted Dropout.
    Srivastava et al. 2014 - Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
    JMLR 2014. https://jmlr.org/papers/v15/srivastava14a.html
    Inverted scaling: divides by (1-p) during training so test-time behavior unchanged.
    """

    def __init__(self, p=0.3):
        self.p = p
        self._mask = None

    def forward(self, x, training=True):
        xp = get_xp(x)
        if training and self.p > 0.0:
            self._mask = (xp.random.random(x.shape) > self.p).astype(x.dtype) / (1.0 - self.p)
            return x * self._mask
        self._mask = None
        return x

    def backward(self, grad):
        if self._mask is not None:
            return grad * self._mask
        return grad

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


class _LinearProjection:
    """1x1 projection shortcut for residual connections with dimension mismatch."""

    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.W = (np.random.randn(in_features, out_features) * scale).astype(np.float32)
        self.b = np.zeros(out_features, dtype=np.float32)
        self.dW = None
        self.db = None

    def to_gpu(self):
        self.W = _to_gpu(self.W)
        self.b = _to_gpu(self.b)

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class PreActivationResidualBlock:
    """
    Pre-activation residual block.
    He et al. 2016 - Identity Mappings in Deep Residual Networks.
    arXiv:1603.05027. https://arxiv.org/abs/1603.05027
    Order: BN -> Act -> Dense -> BN -> Act -> Dropout -> Dense + skip.
    Pre-activation gives cleaner gradient flow than post-activation ResNet.
    Projection shortcut used when in_features != out_features.
    """

    def __init__(self, in_features, out_features, activation, normalization="batch", dropout_rate=0.3):
        from activations import get_activation

        self.in_features = in_features
        self.out_features = out_features

        self._bn1 = make_norm(normalization, in_features)
        self._act1 = get_activation(activation)
        self._dense1 = Dense(in_features, out_features)
        self._bn2 = make_norm(normalization, out_features)
        self._act2 = get_activation(activation)
        self._dropout = Dropout(dropout_rate)
        self._dense2 = Dense(out_features, out_features)

        self._shortcut = None
        if in_features != out_features:
            self._shortcut = _LinearProjection(in_features, out_features)

    def to_gpu(self):
        for layer in [self._bn1, self._dense1, self._bn2, self._dense2]:
            layer.to_gpu()
        if self._shortcut is not None:
            self._shortcut.to_gpu()

    def forward(self, x, training=True):
        self._x_in = x
        out = self._bn1(x, training)
        out = self._act1(out, training)
        out = self._dense1(out, training)
        out = self._bn2(out, training)
        out = self._act2(out, training)
        out = self._dropout(out, training)
        out = self._dense2(out, training)
        skip = self._shortcut.forward(x) if self._shortcut else x
        return out + skip

    def backward(self, grad):
        grad_skip = self._shortcut.backward(grad) if self._shortcut else grad
        g = self._dense2.backward(grad)
        g = self._dropout.backward(g)
        g = self._act2.backward(g)
        g = self._bn2.backward(g)
        g = self._dense1.backward(g)
        g = self._act1.backward(g)
        g = self._bn1.backward(g)
        return g + grad_skip

    def params(self):
        p = (self._bn1.params() + self._dense1.params()
             + self._bn2.params() + self._dense2.params())
        if self._shortcut:
            p += self._shortcut.params()
        return p

    def grads(self):
        g = (self._bn1.grads() + self._dense1.grads()
             + self._bn2.grads() + self._dense2.grads())
        if self._shortcut:
            g += self._shortcut.grads()
        return g

    def __call__(self, x, training=True):
        return self.forward(x, training)


class BottleneckResBlock:
    """
    Bottleneck residual block.
    He et al. 2015 - Deep Residual Learning for Image Recognition.
    arXiv:1512.03385. https://arxiv.org/abs/1512.03385 (Section 4).
    Reduces width to bottleneck_dim then projects back: Dense(B)->BN->Act->Dense(B)->BN->Act->Dense(D)->skip.
    Pre-activation ordering applied for consistency with PreActivationResidualBlock.
    """

    def __init__(self, in_features, bottleneck_dim, activation, normalization="batch", dropout_rate=0.3):
        from activations import get_activation

        self._bn1 = make_norm(normalization, in_features)
        self._act1 = get_activation(activation)
        self._dense1 = Dense(in_features, bottleneck_dim)

        self._bn2 = make_norm(normalization, bottleneck_dim)
        self._act2 = get_activation(activation)
        self._dense2 = Dense(bottleneck_dim, bottleneck_dim)

        self._bn3 = make_norm(normalization, bottleneck_dim)
        self._act3 = get_activation(activation)
        self._dropout = Dropout(dropout_rate)
        self._dense3 = Dense(bottleneck_dim, in_features)

    def to_gpu(self):
        for layer in [self._bn1, self._dense1, self._bn2, self._dense2,
                      self._bn3, self._dense3]:
            layer.to_gpu()

    def forward(self, x, training=True):
        self._x_in = x
        out = self._bn1(x, training)
        out = self._act1(out, training)
        out = self._dense1(out, training)
        out = self._bn2(out, training)
        out = self._act2(out, training)
        out = self._dense2(out, training)
        out = self._bn3(out, training)
        out = self._act3(out, training)
        out = self._dropout(out, training)
        out = self._dense3(out, training)
        return out + x

    def backward(self, grad):
        grad_skip = grad
        g = self._dense3.backward(grad)
        g = self._dropout.backward(g)
        g = self._act3.backward(g)
        g = self._bn3.backward(g)
        g = self._dense2.backward(g)
        g = self._act2.backward(g)
        g = self._bn2.backward(g)
        g = self._dense1.backward(g)
        g = self._act1.backward(g)
        g = self._bn1.backward(g)
        return g + grad_skip

    def params(self):
        return (self._bn1.params() + self._dense1.params()
                + self._bn2.params() + self._dense2.params()
                + self._bn3.params() + self._dense3.params())

    def grads(self):
        return (self._bn1.grads() + self._dense1.grads()
                + self._bn2.grads() + self._dense2.grads()
                + self._bn3.grads() + self._dense3.grads())

    def __call__(self, x, training=True):
        return self.forward(x, training)
