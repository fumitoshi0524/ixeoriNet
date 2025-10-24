package optim

import (
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Adam struct {
	params []*tensor.Tensor
	lr     float64
	beta1  float64
	beta2  float64
	eps    float64
	m      map[*tensor.Tensor]*tensor.Tensor
	v      map[*tensor.Tensor]*tensor.Tensor
	step   int
}

func NewAdam(params []*tensor.Tensor, lr, beta1, beta2, eps float64) *Adam {
	return &Adam{
		params: params,
		lr:     lr,
		beta1:  beta1,
		beta2:  beta2,
		eps:    eps,
		m:      map[*tensor.Tensor]*tensor.Tensor{},
		v:      map[*tensor.Tensor]*tensor.Tensor{},
	}
}

func (o *Adam) Step() error {
	o.step++
	for _, p := range o.params {
		if p == nil {
			continue
		}
		grad := p.Grad()
		if grad == nil {
			continue
		}
		shape := grad.Shape()
		m := o.m[p]
		if m == nil {
			m = tensor.Zeros(shape...)
		}
		v := o.v[p]
		if v == nil {
			v = tensor.Zeros(shape...)
		}
		m.Scale(o.beta1)
		if err := m.AddScaled(grad, 1-o.beta1); err != nil {
			return err
		}
		gradSquared := grad.Clone()
		if err := gradSquared.MulInPlace(grad); err != nil {
			return err
		}
		v.Scale(o.beta2)
		if err := v.AddScaled(gradSquared, 1-o.beta2); err != nil {
			return err
		}
		o.m[p] = m
		o.v[p] = v
		biasCorr1 := 1 - math.Pow(o.beta1, float64(o.step))
		biasCorr2 := 1 - math.Pow(o.beta2, float64(o.step))
		if biasCorr1 == 0 {
			biasCorr1 = math.SmallestNonzeroFloat64
		}
		if biasCorr2 == 0 {
			biasCorr2 = math.SmallestNonzeroFloat64
		}
		mHat := m.Clone()
		mHat.Scale(1 / biasCorr1)
		vHat := v.Clone()
		vHat.Scale(1 / biasCorr2)
		sqrtV := tensor.Pow(vHat, 0.5)
		epsTensor := tensor.Full(o.eps, sqrtV.Shape()...)
		denom, err := tensor.Add(sqrtV, epsTensor)
		if err != nil {
			return err
		}
		update, err := tensor.Div(mHat, denom)
		if err != nil {
			return err
		}
		if err := p.AddScaled(update, -o.lr); err != nil {
			return err
		}
	}
	return nil
}

func (o *Adam) ZeroGrad() {
	for _, p := range o.params {
		if p != nil {
			p.ZeroGrad()
		}
	}
}
