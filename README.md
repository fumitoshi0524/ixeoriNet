# ixeoriNet

`ixeoriNet` is a modular deep-learning library written in Go. It focuses on extensibility and ease of use, featuring automatic differentiation, neural-network building blocks, optimizers, and ready-to-run examples.

## Features

- Autograd-enabled tensor engine with broadcasting, reductions, and in-place operations.
- Neural network modules: linear layers, convolutions (1D/2D/3D), recurrent units (RNN/GRU/LSTM), embeddings, normalization layers, dropout, and pooling.
- Optimizers: SGD (with momentum/Nesterov), Adam, AdamW, RMSProp, Adagrad, Adadelta, plus gradient clipping and parameter constraints.
- Losses: cross-entropy, negative log likelihood, mean squared error.
- Command-line examples, including an MNIST classifier that achieves >97% accuracy.

## Installation

```bash
# Add the module to your project
go get github.com/fumitoshi0524/ixeoriNet
```

The project targets Go 1.21 or newer.

## Repository layout

```
cmd/            Example applications (classifier, demo, mnist)
docs/           Additional documentation (API reference)
internal/       Parallel helpers and utilities
loss/           Loss function implementations
nn/             Neural-network modules
optim/          Optimizers and gradient utilities
tensor/         Tensor data structures and autograd machinery
```

## Running the MNIST examples

```bash
# From the repository root
go run ./cmd/mnist      # MLP classifier
go run ./cmd/mnist_cnn  # CNN classifier
```

Both programs download the MNIST dataset (with multiple mirrors) on first run. The MLP trains for 12 epochs and verifies that the final test accuracy surpasses 97%. The CNN variant trains a small convolutional model with pooling and expects >98.5% test accuracy.

## Testing

```bash
go test ./...
```

The test suite covers tensors, neural-network modules, optimizers, losses, and internal parallel utilities.

## Documentation

See [`docs/API.md`](docs/API.md) for a summarized API reference covering the main packages (`tensor`, `nn`, `optim`, `loss`).

## Contributing

1. Run `go fmt ./...` and `go test ./...` before submitting changes.
2. Keep new components focused and add tests where meaningful.
3. Open an issue if you plan larger architectural changes.

## Aditional Notes

This project originated from a final assignment in a Deep Learning course at UESTC. It may not perform optimally in all scenarios, and its efficiency is still open to improvement. The primary purpose of this repository is to share ideas and code—contributions and suggestions are warmly welcomed. If it proves helpful to you in any way, that would be my greatest pleasure.

## License

[DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE](LICENSE)
