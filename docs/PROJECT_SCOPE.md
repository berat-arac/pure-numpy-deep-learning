# Project Scope

## Included

- neural network layers implemented with NumPy
- manual forward and backward propagation
- CPU training
- real dataset loaders
- optimizers and schedulers
- checkpointing
- gradient validation
- profiling and measured optimization
- vision models from MLP through EfficientNet-B0
- text models from RNN and LSTM through a decoder-only Transformer
- natural-language character completion on small public-domain prose corpora

## Intentionally excluded

- PyTorch
- TensorFlow
- JAX
- CuPy
- automatic differentiation
- CUDA-specific kernels
- distributed training
- ImageNet-scale training as a project requirement
- large-language-model scale training
- production framework compatibility
- Transformer key-value caching

## Reason for the constraint

The NumPy-only rule is the learning and engineering constraint of the project. Replacing NumPy with a GPU-compatible array library would make larger experiments faster, but it would change the central premise of the repository.
