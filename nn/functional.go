package nn

import "github.com/fumitoshi0524/ixeoriNet/tensor"

type Functional struct {
	fn func(*tensor.Tensor) (*tensor.Tensor, error)
}

func NewFunctional(fn func(*tensor.Tensor) (*tensor.Tensor, error)) *Functional {
	return &Functional{fn: fn}
}

func (f *Functional) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return f.fn(input)
}

func (f *Functional) Parameters() []*tensor.Tensor {
	return nil
}

func (f *Functional) ZeroGrad() {}

func Relu() Module {
	return NewFunctional(func(x *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Relu(x), nil
	})
}

func Sigmoid() Module {
	return NewFunctional(func(x *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Sigmoid(x), nil
	})
}

func Tanh() Module {
	return NewFunctional(func(x *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Tanh(x), nil
	})
}
