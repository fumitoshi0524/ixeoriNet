# ixeoriNet API Reference

This document provides a high-level overview of the primary packages exposed by `github.com/fumitoshi0524/ixeoriNet`.

## Package `tensor`

The `tensor` package hosts the autograd-enabled tensor type and fundamental operations.

### Types

- `Tensor`: core data structure storing multidimensional arrays, gradients, and autograd metadata.
- `node`: internal backward graph node (not exported).

### Construction helpers

- `MustNew(data []float64, shape ...int) *Tensor`: create a tensor and panic on invalid shape.
- `Zeros(shape ...int) *Tensor`, `Ones(shape ...int) *Tensor`, `Full(value float64, shape ...int) *Tensor`.
- `Randn(shape ...int) *Tensor`, `Rand(shape ...int) *Tensor`: Gaussian and uniform random tensors (with gradient disabled by default).

### Core functionality

- Autograd control: `SetRequiresGrad(bool)`, `Backward() error`, `ZeroGrad()`, `Detach()`.
- Data manipulation: `Reshape`, `Transpose`, `Permute`, `Flatten`, `Chunk`, `Concat`, `Split`, `Squeeze`, `Unsqueeze`, `Stack`.
- Arithmetic: `Add`, `Sub`, `Mul`, `Div`, broadcasting variations, in-place counterparts (`AddInPlace`, `MulInPlace`, ...).
- Reductions: `Sum`, `Mean`, `LogSumExp`, plus axis-aware versions.
- Neural-ops: `MatMul`, `Linear`, `Conv1D/Conv2D/Conv3D`, pooling (`MaxPool2D`, `AvgPool2D`), activation helpers (`Relu`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`).

Gradients propagate automatically for all operations when operands require gradients. Use `tensor.SaveTensors` / `tensor.LoadTensors` for lightweight checkpointing of parameter maps.

## Package `nn`

`nn` builds modular neural-network layers on top of `tensor`.

### Core concepts

- `Module` interface: defines `Forward(*tensor.Tensor) (*tensor.Tensor, error)`, `Parameters() []*tensor.Tensor`, `ZeroGrad()`.
- `Sequential`: helper to chain multiple modules.
- `StatefulModule`: extends `Module` with serialization hooks (`StateDict` / `LoadState`).

### Modules

- Linear and affine: `NewLinear`.
- Convolutional: `NewConv1d`, `NewConv2d`, `NewConv3d`, and transpose counterparts.
- Recurrent: `NewRNN`, `NewGRU`, `NewLSTM` with configurable input/hidden sizes and layers.
- Embeddings: `NewEmbedding`.
- Normalization: `NewBatchNorm1d/2d/3d`, `NewLayerNorm`.
- Dropout: `NewDropout` (with `Train`/`Eval` toggles).
- Pooling wrappers: `NewMaxPool2d`, `NewAvgPool2d`.
- Functional wrappers: `Relu`, `Sigmoid`, `Tanh` returning `Module` implementations.

All modules expose learnable parameters through `Parameters()` for optimizer registration.

## Package `optim`

Optimizers update parameters using gradients computed through autograd.

### Constructors

- `NewSGD(params []*tensor.Tensor, lr float64, momentum float64)` plus `NewSGDWithConfig` for advanced options (weight decay, Nesterov, gradient clipping, constraints).
- `NewAdam`, `NewAdamW`, `NewRMSProp`, `NewAdagrad`, `NewAdadelta` with algorithm-specific hyperparameters.

### Utilities

- Gradient clipping: `ClipGradNorm`, `ClipGradValue`.
- Constraints: `Constraint` interface with `MaxNormConstraint` implementation.

All optimizers satisfy a minimal interface: `Step() error`, `ZeroGrad()`.

## Package `loss`

Loss functions convert predictions and targets into scalar tensors.

- `CrossEntropy(logits *tensor.Tensor, targets []int)`: multi-class classification loss (averaged over batch).
- `NLL(logProb *tensor.Tensor, targets []int)`: negative log likelihood for log-probability inputs.
- `MSE(pred, target *tensor.Tensor)`: mean squared error for regression tasks.

Loss tensors keep autograd metadata, so calling `lossTensor.Backward()` computes gradients for model parameters.

## Example: training loop

```go
package main

import (
    "github.com/fumitoshi0524/ixeoriNet/loss"
    "github.com/fumitoshi0524/ixeoriNet/nn"
    "github.com/fumitoshi0524/ixeoriNet/optim"
    "github.com/fumitoshi0524/ixeoriNet/tensor"
)

func train(inputs *tensor.Tensor, targets []int) error {
    model := nn.NewSequential(
        nn.NewLinear(784, 256, true),
        nn.Relu(),
        nn.NewLinear(256, 10, true),
    )
    opt := optim.NewAdam(model.Parameters(), 1e-3, 0.9, 0.999, 1e-8)

    opt.ZeroGrad()
    logits, err := model.Forward(inputs)
    if err != nil {
        return err
    }
    ce, err := loss.CrossEntropy(logits, targets)
    if err != nil {
        return err
    }
    if err := ce.Backward(); err != nil {
        return err
    }
    return opt.Step()
}
```

Refer to the package documentation and tests for more detailed usage patterns.
