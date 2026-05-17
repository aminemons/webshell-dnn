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


class ReLU:
    """
    Rectified Linear Unit.
    Nair & Hinton 2010 - Rectified Linear Units Improve Restricted Boltzmann Machines.
    ICML 2010. https://icml.cc/Conferences/2010/papers/432.pdf
    """

    def forward(self, x, training=True):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


class LeakyReLU:
    """
    Leaky Rectified Linear Unit.
    Maas et al. 2013 - Rectifier Nonlinearities Improve Neural Network Acoustic Models.
    ICML 2013. https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
    alpha=0.01 per standard convention and ablation studies.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x, training=True):
        xp = get_xp(x)
        self._mask = x > 0
        return xp.where(self._mask, x, self.alpha * x)

    def backward(self, grad):
        xp = get_xp(grad)
        return xp.where(self._mask, grad, self.alpha * grad)

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


class ELU:
    """
    Exponential Linear Unit.
    Clevert et al. 2015 - Fast and Accurate Deep Network Learning by Exponential Linear Units.
    arXiv:1511.07289. https://arxiv.org/abs/1511.07289
    alpha=1.0 is the value shown in the paper's ablations to yield best results.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x, training=True):
        xp = get_xp(x)
        self._x = x
        self._pos = x >= 0
        return xp.where(self._pos, x, self.alpha * (xp.exp(xp.minimum(x, 0)) - 1.0))

    def backward(self, grad):
        xp = get_xp(grad)
        dx = xp.where(self._pos, xp.ones_like(self._x), self.alpha * xp.exp(xp.minimum(self._x, 0)))
        return grad * dx

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


class GELU:
    """
    Gaussian Error Linear Unit (approximate version).
    Hendrycks & Gimpel 2016 - Gaussian Error Linear Units.
    arXiv:1606.08415. https://arxiv.org/abs/1606.08415
    Approximate: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715*x^3)))
    Used in BERT, GPT-2/3. Empirically outperforms ReLU on many tasks.
    """

    def forward(self, x, training=True):
        xp = get_xp(x)
        self._x = x
        c = xp.sqrt(xp.array(2.0 / np.pi, dtype=x.dtype))
        self._inner = c * (x + 0.044715 * x ** 3)
        self._tanh_val = xp.tanh(self._inner)
        return 0.5 * x * (1.0 + self._tanh_val)

    def backward(self, grad):
        xp = get_xp(grad)
        c = xp.sqrt(xp.array(2.0 / np.pi, dtype=grad.dtype))
        sech2 = 1.0 - self._tanh_val ** 2
        d_inner_dx = c * (1.0 + 3.0 * 0.044715 * self._x ** 2)
        dgelu = 0.5 * (1.0 + self._tanh_val) + 0.5 * self._x * sech2 * d_inner_dx
        return grad * dgelu

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


class Swish:
    """
    Swish / SiLU activation: x * sigmoid(x).
    Ramachandran et al. 2017 - Searching for Activation Functions.
    arXiv:1710.05941. https://arxiv.org/abs/1710.05941
    Found via automated search; frequently matches or exceeds ReLU in deep networks.
    """

    def forward(self, x, training=True):
        xp = get_xp(x)
        self._x = x
        self._sig = 1.0 / (1.0 + xp.exp(-x))
        return x * self._sig

    def backward(self, grad):
        swish_out = self._x * self._sig
        dswish = self._sig + swish_out * (1.0 - self._sig)
        return grad * dswish

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


class Mish:
    """
    Mish activation: x * tanh(softplus(x)).
    Misra 2019 - Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    arXiv:1908.08681. https://arxiv.org/abs/1908.08681
    softplus(x) = log(1 + e^x). Mish is non-monotonic and self-regularizing.
    """

    def forward(self, x, training=True):
        xp = get_xp(x)
        self._x = x
        self._sp = xp.log1p(xp.exp(xp.minimum(x, 20.0)))
        self._tanh_sp = xp.tanh(self._sp)
        return x * self._tanh_sp

    def backward(self, grad):
        xp = get_xp(grad)
        sig = 1.0 / (1.0 + xp.exp(-self._x))
        sech2 = 1.0 - self._tanh_sp ** 2
        dmish = self._tanh_sp + self._x * sech2 * sig
        return grad * dmish

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


class Sigmoid:
    """Sigmoid activation. Used only at output layer for binary classification."""

    def forward(self, x, training=True):
        xp = get_xp(x)
        self._out = 1.0 / (1.0 + xp.exp(-xp.clip(x, -500, 500)))
        return self._out

    def backward(self, grad):
        return grad * self._out * (1.0 - self._out)

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


class Tanh:
    """Hyperbolic tangent activation."""

    def forward(self, x, training=True):
        xp = get_xp(x)
        self._out = xp.tanh(x)
        return self._out

    def backward(self, grad):
        return grad * (1.0 - self._out ** 2)

    def params(self):
        return []

    def grads(self):
        return []

    def __call__(self, x, training=True):
        return self.forward(x, training)


_REGISTRY = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "gelu": GELU,
    "swish": Swish,
    "mish": Mish,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
}

BENCHMARK_ACTIVATIONS = ["relu", "leaky_relu", "elu", "gelu", "swish", "mish"]


def get_activation(name):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()
