package optim

import "github.com/fumitoshi0524/ixeoriNet/tensor"

type SGD struct {
	params        []*tensor.Tensor
	lr            float64
	momentum      float64
	weightDecay   float64
	nesterov      bool
	velocity      map[*tensor.Tensor]*tensor.Tensor
	maxGradNorm   float64
	gradNormType  float64
	gradValueClip float64
	constraints   []Constraint
}

type SGDConfig struct {
	LR            float64
	Momentum      float64
	WeightDecay   float64
	Nesterov      bool
	MaxGradNorm   float64
	GradNormType  float64
	GradValueClip float64
	Constraints   []Constraint
}

func NewSGD(params []*tensor.Tensor, lr float64, momentum float64) *SGD {
	return NewSGDWithConfig(params, SGDConfig{LR: lr, Momentum: momentum})
}

func NewSGDWithConfig(params []*tensor.Tensor, cfg SGDConfig) *SGD {
	vel := make(map[*tensor.Tensor]*tensor.Tensor)
	return &SGD{
		params:        params,
		lr:            cfg.LR,
		momentum:      cfg.Momentum,
		weightDecay:   cfg.WeightDecay,
		nesterov:      cfg.Nesterov,
		velocity:      vel,
		maxGradNorm:   cfg.MaxGradNorm,
		gradNormType:  cfg.GradNormType,
		gradValueClip: cfg.GradValueClip,
		constraints:   append([]Constraint(nil), cfg.Constraints...),
	}
}

func (o *SGD) Step() error {
	if o.maxGradNorm > 0 {
		ClipGradNorm(o.params, o.maxGradNorm, o.gradNormType)
	}
	if o.gradValueClip > 0 {
		ClipGradValue(o.params, o.gradValueClip)
	}
	for _, p := range o.params {
		if p == nil {
			continue
		}
		grad := p.Grad()
		if grad == nil {
			continue
		}
		update := grad
		if o.weightDecay > 0 {
			decaySource := p.Detach()
			if err := update.AddScaled(decaySource, o.weightDecay); err != nil {
				return err
			}
		}
		if o.momentum > 0 {
			v := o.velocity[p]
			if v == nil {
				shape := grad.Shape()
				v = tensor.Zeros(shape...)
			}
			v.Scale(o.momentum)
			if err := v.AddScaled(update, 1.0); err != nil {
				return err
			}
			o.velocity[p] = v
			if o.nesterov {
				tmp := update.Clone()
				if err := tmp.AddScaled(v, o.momentum); err != nil {
					return err
				}
				update = tmp
			} else {
				update = v.Clone()
			}
		}
		if err := p.AddScaled(update, -o.lr); err != nil {
			return err
		}
		for _, c := range o.constraints {
			if err := c.Apply(p); err != nil {
				return err
			}
		}
	}
	return nil
}

func (o *SGD) SetWeightDecay(v float64) {
	o.weightDecay = v
}

func (o *SGD) SetNesterov(enabled bool) {
	o.nesterov = enabled
}

func (o *SGD) WeightDecay() float64 {
	return o.weightDecay
}

func (o *SGD) Nesterov() bool {
	return o.nesterov
}

func (o *SGD) SetGradNorm(maxNorm, normType float64) {
	o.maxGradNorm = maxNorm
	o.gradNormType = normType
}

func (o *SGD) GradNorm() (float64, float64) {
	return o.maxGradNorm, o.gradNormType
}

func (o *SGD) SetGradValueClip(limit float64) {
	o.gradValueClip = limit
}

func (o *SGD) GradValueClip() float64 {
	return o.gradValueClip
}

func (o *SGD) AddConstraint(c Constraint) {
	o.constraints = append(o.constraints, c)
}

func (o *SGD) Constraints() []Constraint {
	return append([]Constraint(nil), o.constraints...)
}

func (o *SGD) ZeroGrad() {
	for _, p := range o.params {
		if p != nil {
			p.ZeroGrad()
		}
	}
}
