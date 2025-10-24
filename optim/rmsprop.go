package optim

import "github.com/fumitoshi0524/ixeoriNet/tensor"

type RMSProp struct {
	params      []*tensor.Tensor
	lr          float64
	alpha       float64
	eps         float64
	weightDecay float64
	momentum    float64
	squareAvg   map[*tensor.Tensor]*tensor.Tensor
	buffer      map[*tensor.Tensor]*tensor.Tensor
}

type RMSPropConfig struct {
	LR          float64
	Alpha       float64
	Eps         float64
	WeightDecay float64
	Momentum    float64
}

func NewRMSProp(params []*tensor.Tensor, lr float64) *RMSProp {
	return NewRMSPropWithConfig(params, RMSPropConfig{LR: lr, Alpha: 0.99, Eps: 1e-8})
}

func NewRMSPropWithConfig(params []*tensor.Tensor, cfg RMSPropConfig) *RMSProp {
	if cfg.Alpha == 0 {
		cfg.Alpha = 0.99
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-8
	}
	return &RMSProp{
		params:      params,
		lr:          cfg.LR,
		alpha:       cfg.Alpha,
		eps:         cfg.Eps,
		weightDecay: cfg.WeightDecay,
		momentum:    cfg.Momentum,
		squareAvg:   map[*tensor.Tensor]*tensor.Tensor{},
		buffer:      map[*tensor.Tensor]*tensor.Tensor{},
	}
}

func (o *RMSProp) Step() error {
	for _, p := range o.params {
		if p == nil {
			continue
		}
		grad := p.Grad()
		if grad == nil {
			continue
		}
		update := grad.Clone()
		if o.weightDecay > 0 {
			if err := update.AddScaled(p, o.weightDecay); err != nil {
				return err
			}
		}
		sq := o.squareAvg[p]
		if sq == nil {
			shape := grad.Shape()
			sq = tensor.Zeros(shape...)
		}
		sq.Scale(o.alpha)
		squared := grad.Clone()
		if err := squared.MulInPlace(grad); err != nil {
			return err
		}
		if err := sq.AddScaled(squared, 1-o.alpha); err != nil {
			return err
		}
		o.squareAvg[p] = sq
		denom := sq.Clone()
		epsTensor := tensor.Full(o.eps, denom.Shape()...)
		var err error
		denom, err = tensor.Add(denom, epsTensor)
		if err != nil {
			return err
		}
		denom = tensor.Pow(denom, 0.5)
		adj, err := tensor.Div(update, denom)
		if err != nil {
			return err
		}
		if o.momentum > 0 {
			buf := o.buffer[p]
			if buf == nil {
				buf = tensor.Zeros(adj.Shape()...)
			}
			buf.Scale(o.momentum)
			if err := buf.AddScaled(adj, o.lr); err != nil {
				return err
			}
			o.buffer[p] = buf
			if err := p.AddScaled(buf, -1); err != nil {
				return err
			}
		} else {
			if err := p.AddScaled(adj, -o.lr); err != nil {
				return err
			}
		}
	}
	return nil
}

func (o *RMSProp) ZeroGrad() {
	for _, p := range o.params {
		if p != nil {
			p.ZeroGrad()
		}
	}
}
