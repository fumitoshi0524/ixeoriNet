package optim

import "math"

import "github.com/fumitoshi0524/ixeoriNet/tensor"

type AdamW struct {
	params      []*tensor.Tensor
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	m           map[*tensor.Tensor]*tensor.Tensor
	v           map[*tensor.Tensor]*tensor.Tensor
	step        int
}

type AdamWConfig struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64
}

func NewAdamW(params []*tensor.Tensor, lr float64) *AdamW {
	return NewAdamWWithConfig(params, AdamWConfig{LR: lr, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8})
}

func NewAdamWWithConfig(params []*tensor.Tensor, cfg AdamWConfig) *AdamW {
	if cfg.Beta1 == 0 {
		cfg.Beta1 = 0.9
	}
	if cfg.Beta2 == 0 {
		cfg.Beta2 = 0.999
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-8
	}
	return &AdamW{
		params:      params,
		lr:          cfg.LR,
		beta1:       cfg.Beta1,
		beta2:       cfg.Beta2,
		eps:         cfg.Eps,
		weightDecay: cfg.WeightDecay,
		m:           map[*tensor.Tensor]*tensor.Tensor{},
		v:           map[*tensor.Tensor]*tensor.Tensor{},
	}
}

func (o *AdamW) Step() error {
	o.step++
	for _, p := range o.params {
		if p == nil {
			continue
		}
		grad := p.Grad()
		if grad == nil {
			continue
		}
		m := o.m[p]
		if m == nil {
			m = tensor.Zeros(grad.Shape()...)
		}
		v := o.v[p]
		if v == nil {
			v = tensor.Zeros(grad.Shape()...)
		}
		m.Scale(o.beta1)
		if err := m.AddScaled(grad, 1-o.beta1); err != nil {
			return err
		}
		sq := grad.Clone()
		if err := sq.MulInPlace(grad); err != nil {
			return err
		}
		v.Scale(o.beta2)
		if err := v.AddScaled(sq, 1-o.beta2); err != nil {
			return err
		}
		o.m[p] = m
		o.v[p] = v
		biasCorr1 := 1 - mathPow(o.beta1, o.step)
		biasCorr2 := 1 - mathPow(o.beta2, o.step)
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
		if o.weightDecay > 0 {
			if err := p.AddScaled(p, -o.lr*o.weightDecay); err != nil {
				return err
			}
		}
		if err := p.AddScaled(update, -o.lr); err != nil {
			return err
		}
	}
	return nil
}

func (o *AdamW) ZeroGrad() {
	for _, p := range o.params {
		if p != nil {
			p.ZeroGrad()
		}
	}
}

func mathPow(base float64, step int) float64 {
	result := 1.0
	for i := 0; i < step; i++ {
		result *= base
	}
	return result
}
