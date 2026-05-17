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


class SGDNesterov:
    """
    SGD with Nesterov Accelerated Gradient.
    Nesterov 1983 - A Method of Solving a Convex Programming Problem with Convergence Rate O(1/k^2).
    Soviet Mathematics Doklady 27(2):372-376. 1983.
    Sutskever et al. 2013 formulation: look-ahead gradient, then momentum step.
    v_t = mu * v_{t-1} - lr * grad(theta + mu * v_{t-1})
    theta = theta + v_t
    """

    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._v = None

    def step(self, params, grads):
        if self._v is None:
            self._v = [get_xp(p).zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * p
            v_prev = self._v[i].copy()
            self._v[i] = self.momentum * self._v[i] - self.lr * g
            p += -self.momentum * v_prev + (1.0 + self.momentum) * self._v[i]


class Adam:
    """
    Adam optimizer with bias correction.
    Kingma & Ba 2014 - Adam: A Method for Stochastic Optimization.
    arXiv:1412.6980. https://arxiv.org/abs/1412.6980
    beta1=0.9, beta2=0.999, eps=1e-8 are the values from the paper.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params, grads):
        if self._m is None:
            self._m = [get_xp(p).zeros_like(p) for p in params]
            self._v = [get_xp(p).zeros_like(p) for p in params]

        self._t += 1
        bias_corr1 = 1.0 - self.beta1 ** self._t
        bias_corr2 = 1.0 - self.beta2 ** self._t
        lr_t = self.lr * np.sqrt(bias_corr2) / bias_corr1

        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            xp = get_xp(p)
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * p
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g ** 2
            p -= lr_t * self._m[i] / (xp.sqrt(self._v[i]) + self.eps)


class AdamW:
    """
    Adam with decoupled weight decay (AdamW).
    Loshchilov & Hutter 2019 - Decoupled Weight Decay Regularization.
    arXiv:1711.05101. https://arxiv.org/abs/1711.05101
    Weight decay applied directly to parameters, NOT folded into the gradient.
    This corrects the L2 regularization conflation in vanilla Adam.
    lambda=0.01 per paper recommendation.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = None
        self._v = None
        self._t = 0

    def step(self, params, grads):
        if self._m is None:
            self._m = [get_xp(p).zeros_like(p) for p in params]
            self._v = [get_xp(p).zeros_like(p) for p in params]

        self._t += 1
        bias_corr1 = 1.0 - self.beta1 ** self._t
        bias_corr2 = 1.0 - self.beta2 ** self._t
        lr_t = self.lr * np.sqrt(bias_corr2) / bias_corr1

        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            xp = get_xp(p)
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g ** 2
            p -= lr_t * self._m[i] / (xp.sqrt(self._v[i]) + self.eps)
            p -= self.lr * self.weight_decay * p


class RAdam:
    """
    Rectified Adam.
    Liu et al. 2019 - On the Variance of the Adaptive Learning Rate and Beyond.
    arXiv:1908.03265. https://arxiv.org/abs/1908.03265
    Computes variance rectification term rho_t. When rho_t > 4, applies full
    adaptive step; otherwise falls back to SGD-like update. Eliminates warmup.
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = None
        self._v = None
        self._t = 0
        self._rho_inf = 2.0 / (1.0 - beta2) - 1.0

    def step(self, params, grads):
        if self._m is None:
            self._m = [get_xp(p).zeros_like(p) for p in params]
            self._v = [get_xp(p).zeros_like(p) for p in params]

        self._t += 1
        beta1_t = self.beta1 ** self._t
        beta2_t = self.beta2 ** self._t
        rho_t = self._rho_inf - 2.0 * self._t * beta2_t / (1.0 - beta2_t)

        for i, (p, g) in enumerate(zip(params, grads)):
            if g is None:
                continue
            xp = get_xp(p)
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * p
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g ** 2
            m_hat = self._m[i] / (1.0 - beta1_t)
            if rho_t > 4.0:
                v_hat = xp.sqrt(self._v[i] / (1.0 - beta2_t))
                rect = float(np.sqrt(
                    ((rho_t - 4.0) * (rho_t - 2.0) * self._rho_inf)
                    / ((self._rho_inf - 4.0) * (self._rho_inf - 2.0) * rho_t)
                ))
                p -= self.lr * rect * m_hat / (v_hat + self.eps)
            else:
                p -= self.lr * m_hat


class Lookahead:
    """
    Lookahead optimizer wrapper.
    Zhang et al. 2019 - Lookahead Optimizer: k steps forward, 1 step back.
    arXiv:1907.08610. https://arxiv.org/abs/1907.08610
    Maintains slow weights updated every k inner steps via linear interpolation.
    Lookahead(RAdam) = Ranger. k=5, alpha=0.5 per paper.
    """

    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_count = 0
        self._slow = None

    @property
    def lr(self):
        return self.base_optimizer.lr

    @lr.setter
    def lr(self, value):
        self.base_optimizer.lr = value

    def step(self, params, grads):
        if self._slow is None:
            self._slow = [p.copy() for p in params]

        self.base_optimizer.step(params, grads)
        self._step_count += 1

        if self._step_count % self.k == 0:
            for i, (slow, fast) in enumerate(zip(self._slow, params)):
                slow += self.alpha * (fast - slow)
                fast[...] = slow
                self._slow[i] = slow.copy()


class LinearWarmup:
    """
    Linear learning rate warmup.
    Goyal et al. 2017 - Accurate, Large Minibatch SGD (warmup heuristic).
    arXiv:1706.02677. https://arxiv.org/abs/1706.02677
    Ramps lr from 0 to lr_max over warmup_epochs.
    """

    def __init__(self, optimizer, warmup_epochs, lr_max):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.lr_max = lr_max
        self.optimizer.lr = 0.0

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.lr_max * (epoch + 1) / self.warmup_epochs
        else:
            lr = self.lr_max
        self.optimizer.lr = lr
        return lr


class CosineAnnealingWarmRestarts:
    """
    Cosine Annealing with Warm Restarts (SGDR).
    Loshchilov & Hutter 2017 - SGDR: Stochastic Gradient Descent with Warm Restarts.
    arXiv:1608.03983. https://arxiv.org/abs/1608.03983
    T_0=200 epochs, T_mult=2, lr_min=1e-6 per paper recommendations.
    """

    def __init__(self, optimizer, lr_min, lr_max, T_0=200, T_mult=2, warmup_epochs=50):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.T_0 = T_0
        self.T_mult = T_mult
        self.warmup_epochs = warmup_epochs

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.lr_max * (epoch + 1) / self.warmup_epochs
            self.optimizer.lr = lr
            return lr
        e = epoch - self.warmup_epochs
        T_cur = e
        T_i = self.T_0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i = int(T_i * self.T_mult)
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + np.cos(np.pi * T_cur / T_i))
        self.optimizer.lr = lr
        return lr


class OneCycleLR:
    """
    One-cycle learning rate policy (super-convergence).
    Smith & Topin 2018 - Super-Convergence: Very Fast Training using Large Learning Rates.
    arXiv:1708.07120. https://arxiv.org/abs/1708.07120
    Phase 1 (warmup to lr_max, 45% of total): linear increase.
    Phase 2 (lr_max to lr_min, 45% of total): cosine decrease.
    Phase 3 (final annealing, 10%): drop to lr_min/1e4.
    """

    def __init__(self, optimizer, lr_max, total_epochs, lr_min_factor=0.01, pct_start=0.45):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_max * lr_min_factor
        self.total_epochs = total_epochs
        self.pct_start = pct_start
        self._p1 = int(total_epochs * pct_start)
        self._p2 = int(total_epochs * (1.0 - 0.10))

    def step(self, epoch):
        if epoch <= self._p1:
            t = epoch / self._p1
            lr = self.lr_min + (self.lr_max - self.lr_min) * t
        elif epoch <= self._p2:
            t = (epoch - self._p1) / (self._p2 - self._p1)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + np.cos(np.pi * t))
        else:
            t = (epoch - self._p2) / (self.total_epochs - self._p2)
            lr = self.lr_min * (1.0 - t) + (self.lr_min * 1e-4) * t
        self.optimizer.lr = lr
        return lr


def make_optimizer(name, lr=1e-3, weight_decay=0.01):
    if name == "sgd":
        return SGDNesterov(lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == "adam":
        return Adam(lr=lr, weight_decay=0.0)
    elif name == "adamw":
        return AdamW(lr=lr, weight_decay=weight_decay)
    elif name == "radam":
        return RAdam(lr=lr, weight_decay=0.0)
    elif name == "ranger":
        base = RAdam(lr=lr, weight_decay=0.0)
        return Lookahead(base, k=5, alpha=0.5)
    else:
        raise ValueError(f"Unknown optimizer '{name}'")


def make_scheduler(optimizer, scheduler_type, lr_max, total_epochs, warmup_epochs=50):
    if scheduler_type == "cosine":
        return CosineAnnealingWarmRestarts(
            optimizer, lr_min=1e-6, lr_max=lr_max,
            T_0=200, T_mult=2, warmup_epochs=warmup_epochs
        )
    elif scheduler_type == "onecycle":
        return OneCycleLR(optimizer, lr_max=lr_max, total_epochs=total_epochs)
    else:
        raise ValueError(f"Unknown scheduler '{scheduler_type}'")
