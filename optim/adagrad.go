package optim

import (
	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Adagrad struct {
	params     []*tensor.Tensor
	lr         float64
	eps        float64
	sumSquares map[*tensor.Tensor]*tensor.Tensor
}

func NewAdagrad(params []*tensor.Tensor, lr, eps float64) *Adagrad {
	if eps <= 0 {
		eps = 1e-10
	}
	return &Adagrad{
		params:     params,
		lr:         lr,
		eps:        eps,
		sumSquares: make(map[*tensor.Tensor]*tensor.Tensor),
	}
}

func (o *Adagrad) Step() error {
	for _, p := range o.params {
		if p == nil {
			continue
		}
		grad := p.Grad()
		if grad == nil {
			continue
		}
		sum := o.sumSquares[p]
		if sum == nil {
			sum = tensor.Zeros(grad.Shape()...)
		}
		gradSquared := grad.Clone()
		if err := gradSquared.MulInPlace(grad); err != nil {
			return err
		}
		if err := sum.AddScaled(gradSquared, 1.0); err != nil {
			return err
		}
		o.sumSquares[p] = sum
		sqrtSum := tensor.Pow(sum, 0.5)
		epsTensor := tensor.Full(o.eps, sqrtSum.Shape()...)
		denom, err := tensor.Add(sqrtSum, epsTensor)
		if err != nil {
			return err
		}
		update, err := tensor.Div(grad, denom)
		if err != nil {
			return err
		}
		if err := p.AddScaled(update, -o.lr); err != nil {
			return err
		}
	}
	return nil
}

func (o *Adagrad) ZeroGrad() {
	for _, p := range o.params {
		if p != nil {
			p.ZeroGrad()
		}
	}
}
