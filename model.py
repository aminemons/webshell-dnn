import os
import numpy as np

try:
    import cupy as cp
    _GPU_AVAILABLE = True
except ImportError:
    cp = None
    _GPU_AVAILABLE = False

from layers import Dense, BatchNormalization, LayerNormalization, Dropout, PreActivationResidualBlock, BottleneckResBlock
from activations import get_activation
from optimizers import make_optimizer, make_scheduler


def get_xp(arr):
    if _GPU_AVAILABLE and cp is not None:
        return cp.get_array_module(arr)
    return np


def _sigmoid(x):
    xp = get_xp(x)
    return 1.0 / (1.0 + xp.exp(-xp.clip(x, -500, 500)))


def _bce_loss(logits, targets):
    xp = get_xp(logits)
    p = _sigmoid(logits)
    t = targets[:, None] if targets.ndim == 1 else targets
    t = xp.array(t, dtype=logits.dtype)
    loss = -(t * xp.log(p + 1e-8) + (1.0 - t) * xp.log(1.0 - p + 1e-8))
    return float(loss.mean()), p


def _clip_gradients(grads, max_norm=1.0):
    """
    Global gradient norm clipping.
    Pascanu et al. 2013 - On the Difficulty of Training Recurrent Neural Networks.
    arXiv:1211.5063. https://arxiv.org/abs/1211.5063 (Algorithm 1).
    """
    filtered = [g for g in grads if g is not None]
    if not filtered:
        return grads
    xp = get_xp(filtered[0])
    total_norm = xp.sqrt(sum(xp.sum(g ** 2) for g in filtered))
    total_norm_scalar = float(total_norm)
    if total_norm_scalar > max_norm:
        coef = max_norm / (total_norm_scalar + 1e-8)
        return [g * coef if g is not None else None for g in grads]
    return grads


class Sequential:
    """Container that chains layers sequentially for forward and backward passes."""

    def __init__(self, layers):
        self._layers = layers

    def to_gpu(self):
        for layer in self._layers:
            if hasattr(layer, "to_gpu"):
                layer.to_gpu()

    def forward(self, x, training=True):
        for layer in self._layers:
            x = layer(x, training)
        return x

    def backward(self, grad):
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        p = []
        for layer in self._layers:
            p += layer.params()
        return p

    def grads(self):
        g = []
        for layer in self._layers:
            g += layer.grads()
        return g

    def get_layer_grad_norms(self):
        norms = []
        for layer in self._layers:
            g_list = layer.grads()
            if g_list:
                xp = get_xp(g_list[0])
                norm = float(xp.sqrt(sum(xp.sum(g ** 2) for g in g_list if g is not None)))
                norms.append(norm)
        return norms

    def save(self, path):
        params = self.params()
        arrays = {}
        for i, p in enumerate(params):
            if _GPU_AVAILABLE and cp is not None and isinstance(p, cp.ndarray):
                arrays[f"p{i}"] = cp.asnumpy(p)
            else:
                arrays[f"p{i}"] = p
        np.savez(path, **arrays)

    def load(self, path):
        data = np.load(path + ".npz" if not path.endswith(".npz") else path)
        params = self.params()
        for i, p in enumerate(params):
            key = f"p{i}"
            if key in data:
                if _GPU_AVAILABLE and cp is not None and isinstance(p, cp.ndarray):
                    p[...] = cp.asarray(data[key])
                else:
                    p[...] = data[key]


def build_model(input_dim, hidden_dim, num_blocks, activation, dropout_rate,
                use_residual, normalization, arch="B"):
    """
    Model factory: constructs network from spec.

    arch='A': plain 40-layer dense network (no residual connections).
    arch='B': pre-activation residual blocks (He et al. 2016).
    arch='C': bottleneck residual blocks (He et al. 2015).
    """
    layers = []

    if arch == "A":
        layers.append(Dense(input_dim, hidden_dim))
        if normalization == "batch":
            layers.append(BatchNormalization(hidden_dim))
        else:
            layers.append(LayerNormalization(hidden_dim))
        layers.append(get_activation(activation))
        layers.append(Dropout(dropout_rate))
        for _ in range(num_blocks - 1):
            layers.append(Dense(hidden_dim, hidden_dim))
            if normalization == "batch":
                layers.append(BatchNormalization(hidden_dim))
            else:
                layers.append(LayerNormalization(hidden_dim))
            layers.append(get_activation(activation))
            layers.append(Dropout(dropout_rate))
        layers.append(Dense(hidden_dim, 1))

    elif arch == "B":
        layers.append(Dense(input_dim, hidden_dim))
        for _ in range(num_blocks):
            layers.append(
                PreActivationResidualBlock(
                    hidden_dim, hidden_dim, activation,
                    normalization=normalization, dropout_rate=dropout_rate
                )
            )
        layers.append(Dense(hidden_dim, 64))
        if normalization == "batch":
            layers.append(BatchNormalization(64))
        else:
            layers.append(LayerNormalization(64))
        layers.append(get_activation(activation))
        layers.append(Dropout(dropout_rate))
        layers.append(Dense(64, 1))

    elif arch == "C":
        layers.append(Dense(input_dim, hidden_dim))
        for _ in range(num_blocks):
            layers.append(
                BottleneckResBlock(
                    hidden_dim, 64, activation,
                    normalization=normalization, dropout_rate=dropout_rate
                )
            )
        layers.append(Dense(hidden_dim, 1))

    else:
        raise ValueError(f"Unknown arch '{arch}'")

    return Sequential(layers)


class Trainer:
    """Full training loop with early stopping, gradient clipping, and checkpointing."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        batch_size=64,
        max_epochs=1500,
        patience=150,
        grad_clip=1.0,
        token_dropout_p=0.05,
        checkpoint_path="best_model",
        verbose=True,
        log_every=10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.token_dropout_p = token_dropout_p
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.log_every = log_every

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "lr": [], "grad_norms": {}
        }
        self._best_val_loss = np.inf
        self._patience_counter = 0
        self._grad_norm_epochs = {1, 100, 500, 1000}

    def _to_gpu(self, X):
        if _GPU_AVAILABLE and cp is not None:
            return cp.asarray(X)
        return X

    def _to_cpu(self, X):
        if _GPU_AVAILABLE and cp is not None and isinstance(X, cp.ndarray):
            return cp.asnumpy(X)
        return X

    def _apply_token_dropout(self, X, rng):
        if self.token_dropout_p <= 0.0:
            return X
        xp = get_xp(X)
        mask = xp.array(rng.random(X.shape) > self.token_dropout_p, dtype=X.dtype)
        return X * mask

    def _run_epoch(self, X, y, training):
        xp = get_xp(X)
        n = X.shape[0]
        total_loss = 0.0
        total_correct = 0
        steps = 0
        rng = np.random.RandomState(0)

        if training:
            perm = xp.array(np.random.permutation(n))
        else:
            perm = xp.arange(n)

        for start in range(0, n, self.batch_size):
            idx = perm[start: start + self.batch_size]
            Xb = X[idx]
            yb = y[idx] if not isinstance(y, np.ndarray) else xp.array(y[idx])

            if training:
                Xb = self._apply_token_dropout(Xb, rng)

            logits = self.model.forward(Xb, training=training)
            loss, p = _bce_loss(logits, yb)
            total_loss += loss
            steps += 1

            pred = (p >= 0.5).astype(xp.int32).reshape(-1)
            yb_flat = yb.reshape(-1).astype(xp.int32)
            total_correct += int((pred == yb_flat).sum())

            if training:
                grad = (p - yb[:, None].astype(logits.dtype)) / Xb.shape[0]
                self.model.backward(grad)
                raw_grads = self.model.grads()
                clipped_grads = _clip_gradients(raw_grads, self.grad_clip)
                params = self.model.params()
                self.optimizer.step(params, clipped_grads)

        avg_loss = total_loss / steps
        avg_acc = total_correct / n
        return avg_loss, avg_acc

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = self._to_gpu(X_train)
        X_val = self._to_gpu(X_val)
        y_train_gpu = self._to_gpu(y_train)
        y_val_gpu = self._to_gpu(y_val)

        for epoch in range(self.max_epochs):
            current_lr = self.scheduler.step(epoch) if self.scheduler else self.optimizer.lr

            train_loss, train_acc = self._run_epoch(X_train, y_train_gpu, training=True)
            val_loss, val_acc = self._run_epoch(X_val, y_val_gpu, training=False)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            if (epoch + 1) in self._grad_norm_epochs:
                self.history["grad_norms"][epoch + 1] = self.model.get_layer_grad_norms()

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
                self.model.save(self.checkpoint_path)
            else:
                self._patience_counter += 1

            if self.verbose and ((epoch + 1) % self.log_every == 0 or epoch == 0):
                print(
                    f"Epoch {epoch+1:5d} | lr={current_lr:.6f} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

            if self._patience_counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1} (patience={self.patience})")
                break

        if os.path.exists(self.checkpoint_path + ".npz"):
            self.model.load(self.checkpoint_path + ".npz")

        return self.history

    def predict_proba(self, X):
        X_gpu = self._to_gpu(X)
        n = X_gpu.shape[0]
        probs = []
        for start in range(0, n, self.batch_size):
            Xb = X_gpu[start: start + self.batch_size]
            logits = self.model.forward(Xb, training=False)
            p = _sigmoid(logits)
            probs.append(self._to_cpu(p).reshape(-1))
        return np.concatenate(probs)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(np.int32)


def build_and_train(
    X_train, y_train, X_val, y_val,
    arch="B",
    hidden_dim=256,
    num_blocks=20,
    activation="gelu",
    dropout_rate=0.3,
    normalization="batch",
    optimizer_name="adamw",
    lr=1e-3,
    weight_decay=0.01,
    scheduler_type="cosine",
    max_epochs=1500,
    patience=150,
    batch_size=64,
    checkpoint_path="results/best_model",
    verbose=True,
    log_every=10,
):
    input_dim = X_train.shape[1]
    model = build_model(
        input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks,
        activation=activation, dropout_rate=dropout_rate,
        use_residual=(arch in ("B", "C")), normalization=normalization, arch=arch,
    )
    if _GPU_AVAILABLE:
        model.to_gpu()

    optimizer = make_optimizer(optimizer_name, lr=lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, scheduler_type, lr_max=lr, total_epochs=max_epochs)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        batch_size=batch_size, max_epochs=max_epochs, patience=patience,
        grad_clip=1.0, token_dropout_p=0.05,
        checkpoint_path=checkpoint_path, verbose=verbose, log_every=log_every,
    )
    history = trainer.fit(X_train, y_train, X_val, y_val)
    return trainer, history
