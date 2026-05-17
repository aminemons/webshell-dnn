# References

Every non-trivial technique implemented in this project is cited here.

---

## Vectorization

### Byte Pair Encoding (BPE)
- **Paper**: Neural Machine Translation of Rare Words with Subword Units
- **Authors**: Rico Sennrich, Barry Haddow, Alexandra Birch
- **Year**: 2016
- **URL**: https://arxiv.org/abs/1508.07909
- **Used in**: `vectorization.py` — `BPETokenizer` class
- **Key implementation detail**: Character-level BPE with end-of-word marker `</w>`. Merge table trained on training corpus only (no leakage). Top-256 subwords selected by document frequency for vocabulary.

### Sublinear TF-IDF Scaling
- **Paper**: A Statistical Interpretation of Term Specificity and Its Application in Retrieval
- **Authors**: Karen Sparck Jones
- **Year**: 1972
- **URL**: https://doi.org/10.1108/eb026526
- **Used in**: `vectorization.py` — `TFIDFVectorizer`
- **Key implementation detail**: Sublinear TF: tf(t,d) = 1 + log(count) to dampen high-frequency token dominance.

### Smoothed IDF
- **Paper**: Term-Weighting Approaches in Automatic Text Retrieval
- **Authors**: Gerard Salton, Christopher Buckley
- **Year**: 1988
- **URL**: https://doi.org/10.1016/0306-4573(88)90021-0
- **Used in**: `vectorization.py` — `TFIDFVectorizer`
- **Key implementation detail**: idf(t) = log(N / df(t)) + 1. Smoothing prevents zero IDF for corpus-wide tokens.

---

## Layers

### He Initialization
- **Paper**: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year**: 2015
- **URL**: https://arxiv.org/abs/1502.01852
- **Used in**: `layers.py` — `Dense.__init__`
- **Key implementation detail**: W ~ N(0, sqrt(2/n_in)). Derived for ReLU-like activations. Applied to all Dense layers.

### Batch Normalization
- **Paper**: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- **Authors**: Sergey Ioffe, Christian Szegedy
- **Year**: 2015
- **URL**: https://arxiv.org/abs/1502.03167
- **Used in**: `layers.py` — `BatchNormalization`
- **Key implementation detail**: Numerically stable backward using the analytical formulation from the paper's appendix. Running mean/variance with momentum=0.9 for inference.

### Layer Normalization
- **Paper**: Layer Normalization
- **Authors**: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
- **Year**: 2016
- **URL**: https://arxiv.org/abs/1607.06450
- **Used in**: `layers.py` — `LayerNormalization`
- **Key implementation detail**: Normalizes across feature dimension (axis=-1) per sample, not across batch. More stable for small batch sizes.

### Dropout
- **Paper**: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- **Authors**: Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
- **Year**: 2014
- **URL**: https://jmlr.org/papers/v15/srivastava14a.html
- **Used in**: `layers.py` — `Dropout`
- **Key implementation detail**: Inverted dropout — scale by 1/(1-p) during training so no test-time correction needed. New binary mask sampled each forward pass.

### Deep Residual Networks
- **Paper**: Deep Residual Learning for Image Recognition
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year**: 2015
- **URL**: https://arxiv.org/abs/1512.03385
- **Used in**: `layers.py` — `BottleneckResBlock`
- **Key implementation detail**: F(x) + x skip connection. Bottleneck design: compress to 64 dims, process, expand back to 256. Enables training very deep networks.

### Pre-activation Residual Connections
- **Paper**: Identity Mappings in Deep Residual Networks
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Year**: 2016
- **URL**: https://arxiv.org/abs/1603.05027
- **Used in**: `layers.py` — `PreActivationResidualBlock`
- **Key implementation detail**: Order BN→Act→Dense→BN→Act→Dense + skip. Pre-activation gives cleaner gradient signal and better regularization than post-activation.

---

## Activations

### ReLU
- **Paper**: Rectified Linear Units Improve Restricted Boltzmann Machines
- **Authors**: Vinod Nair, Geoffrey E. Hinton
- **Year**: 2010
- **URL**: https://icml.cc/Conferences/2010/papers/432.pdf
- **Used in**: `activations.py` — `ReLU`
- **Key implementation detail**: max(0, x). Binary mask cached for backward pass.

### Leaky ReLU
- **Paper**: Rectifier Nonlinearities Improve Neural Network Acoustic Models
- **Authors**: Andrew Maas, Awni Hannun, Andrew Ng
- **Year**: 2013
- **URL**: https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
- **Used in**: `activations.py` — `LeakyReLU`
- **Key implementation detail**: alpha=0.01 (standard). Avoids dead neurons by allowing small negative gradient.

### ELU
- **Paper**: Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
- **Authors**: Djork-Arne Clevert, Thomas Unterthiner, Sepp Hochreiter
- **Year**: 2015
- **URL**: https://arxiv.org/abs/1511.07289
- **Used in**: `activations.py` — `ELU`
- **Key implementation detail**: alpha=1.0 per paper ablation. Exponential clipped at 0 to avoid overflow.

### GELU
- **Paper**: Gaussian Error Linear Units (GELUs)
- **Authors**: Dan Hendrycks, Kevin Gimpel
- **Year**: 2016
- **URL**: https://arxiv.org/abs/1606.08415
- **Used in**: `activations.py` — `GELU`
- **Key implementation detail**: Approximate formulation: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))). Used in BERT, GPT. Derivative computed analytically via chain rule through tanh.

### Swish / SiLU
- **Paper**: Searching for Activation Functions
- **Authors**: Prajit Ramachandran, Barret Zoph, Quoc V. Le
- **Year**: 2017
- **URL**: https://arxiv.org/abs/1710.05941
- **Used in**: `activations.py` — `Swish`
- **Key implementation detail**: x * sigmoid(x). Derivative: sigmoid(x) + swish(x)(1 - sigmoid(x)).

### Mish
- **Paper**: Mish: A Self Regularized Non-Monotonic Neural Activation Function
- **Authors**: Diganta Misra
- **Year**: 2019
- **URL**: https://arxiv.org/abs/1908.08681
- **Used in**: `activations.py` — `Mish`
- **Key implementation detail**: x * tanh(softplus(x)). Non-monotonic and self-regularizing. softplus clamped at 20 to prevent overflow.

---

## Optimizers

### Nesterov Accelerated Gradient
- **Paper**: A Method of Solving a Convex Programming Problem with Convergence Rate O(1/k^2)
- **Authors**: Yurii Nesterov
- **Year**: 1983
- **URL**: https://doi.org/10.1007/978-3-319-91578-4_2 (English translation)
- **Used in**: `optimizers.py` — `SGDNesterov`
- **Key implementation detail**: Look-ahead gradient evaluation. Sutskever et al. 2013 reformulation: v_t = mu*v_{t-1} - lr*g; theta += -mu*v_{t-1} + (1+mu)*v_t.

### Adam
- **Paper**: Adam: A Method for Stochastic Optimization
- **Authors**: Diederik P. Kingma, Jimmy Ba
- **Year**: 2014
- **URL**: https://arxiv.org/abs/1412.6980
- **Used in**: `optimizers.py` — `Adam`
- **Key implementation detail**: Bias-corrected first/second moment estimates. beta1=0.9, beta2=0.999, eps=1e-8 from paper.

### AdamW
- **Paper**: Decoupled Weight Decay Regularization
- **Authors**: Ilya Loshchilov, Frank Hutter
- **Year**: 2019 (ICLR)
- **URL**: https://arxiv.org/abs/1711.05101
- **Used in**: `optimizers.py` — `AdamW`
- **Key implementation detail**: Decoupled weight decay: parameter updated separately from adaptive moment, NOT added to gradient. lambda=0.01 per recommendation.

### RAdam
- **Paper**: On the Variance of the Adaptive Learning Rate and Beyond
- **Authors**: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, Jiawei Han
- **Year**: 2019
- **URL**: https://arxiv.org/abs/1908.03265
- **Used in**: `optimizers.py` — `RAdam`
- **Key implementation detail**: Variance rectification term rho_t. When rho_t > 4, full adaptive step; else SGD-like update. rho_inf = 2/(1-beta2) - 1.

### Lookahead (Ranger)
- **Paper**: Lookahead Optimizer: k steps forward, 1 step back
- **Authors**: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba
- **Year**: 2019
- **URL**: https://arxiv.org/abs/1907.08610
- **Used in**: `optimizers.py` — `Lookahead` (wrapping `RAdam` = Ranger)
- **Key implementation detail**: Slow weights updated every k=5 steps: slow += alpha*(fast-slow). alpha=0.5. Reduces variance and improves generalization.

---

## Learning Rate Scheduling

### SGDR: Cosine Annealing with Warm Restarts
- **Paper**: SGDR: Stochastic Gradient Descent with Warm Restarts
- **Authors**: Ilya Loshchilov, Frank Hutter
- **Year**: 2017 (ICLR)
- **URL**: https://arxiv.org/abs/1608.03983
- **Used in**: `optimizers.py` — `CosineAnnealingWarmRestarts`
- **Key implementation detail**: T_0=200 epochs, T_mult=2. lr_min=1e-6. Warm restarts help escape local minima.

### One-Cycle Policy / Super-Convergence
- **Paper**: Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates
- **Authors**: Leslie N. Smith, Nicholay Topin
- **Year**: 2018
- **URL**: https://arxiv.org/abs/1708.07120
- **Used in**: `optimizers.py` — `OneCycleLR`
- **Key implementation detail**: Phase 1 (45%): linear warmup to lr_max. Phase 2 (45%): cosine annealing to lr_min. Phase 3 (10%): final annealing to lr_min/1e4.

### Linear Warmup
- **Paper**: Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
- **Authors**: Priya Goyal et al.
- **Year**: 2017
- **URL**: https://arxiv.org/abs/1706.02677
- **Used in**: `optimizers.py` — `LinearWarmup` (integrated in `CosineAnnealingWarmRestarts`)
- **Key implementation detail**: Ramp lr from 0 to lr_max over 50 warmup epochs to stabilize early training.

---

## Training

### Gradient Clipping by Global Norm
- **Paper**: On the Difficulty of Training Recurrent Neural Networks
- **Authors**: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio
- **Year**: 2013
- **URL**: https://arxiv.org/abs/1211.5063
- **Used in**: `model.py` — `_clip_gradients`
- **Key implementation detail**: global_norm = sqrt(sum ||g_i||^2). If global_norm > threshold, scale all gradients by threshold/global_norm. Threshold=1.0.

---

## Visualization

### t-SNE
- **Paper**: Visualizing Data using t-SNE
- **Authors**: Laurens van der Maaten, Geoffrey Hinton
- **Year**: 2008
- **URL**: https://www.jmlr.org/papers/v9/vandermaaten08a.html
- **Used in**: `visualization.py` — `_tsne_2d`
- **Key implementation detail**: PCA initialization for stability. Early exaggeration (4x) for first 100 iterations. Perplexity=30. Gradient descent with adaptive gain and momentum.
