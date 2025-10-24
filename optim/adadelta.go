package optim

import "github.com/fumitoshi0524/ixeoriNet/tensor"

type Adadelta struct {
	params    []*tensor.Tensor
	rho       float64
	eps       float64
	lr        float64
	squareAvg map[*tensor.Tensor]*tensor.Tensor
	deltaAvg  map[*tensor.Tensor]*tensor.Tensor
}

func NewAdadelta(params []*tensor.Tensor, lr, rho, eps float64) *Adadelta {
	if rho <= 0 || rho >= 1 {
		rho = 0.9
	}
	if eps <= 0 {
		eps = 1e-6
	}
	if lr <= 0 {
		lr = 1.0
	}
	return &Adadelta{
		params:    params,
		rho:       rho,
		eps:       eps,
		lr:        lr,
		squareAvg: make(map[*tensor.Tensor]*tensor.Tensor),
		deltaAvg:  make(map[*tensor.Tensor]*tensor.Tensor),
	}
}

func (o *Adadelta) Step() error {
	for _, p := range o.params {
		if p == nil {
			continue
		}
		grad := p.Grad()
		if grad == nil {
			continue
		}
		sqAvg := o.squareAvg[p]
		if sqAvg == nil {
			sqAvg = tensor.Zeros(grad.Shape()...)
		}
		dAvg := o.deltaAvg[p]
		if dAvg == nil {
			dAvg = tensor.Zeros(grad.Shape()...)
		}
		gradSquared := grad.Clone()
		if err := gradSquared.MulInPlace(grad); err != nil {
			return err
		}
		sqAvg.Scale(o.rho)
		if err := sqAvg.AddScaled(gradSquared, 1-o.rho); err != nil {
			return err
		}
		rmsGrad := tensor.Pow(sqAvg, 0.5)
		rmsDelta := tensor.Pow(dAvg, 0.5)
		epsTensor := tensor.Full(o.eps, rmsGrad.Shape()...)
		rmsGrad, err := tensor.Add(rmsGrad, epsTensor)
		if err != nil {
			return err
		}
		rmsDelta, err = tensor.Add(rmsDelta, epsTensor)
		if err != nil {
			return err
		}
		ratio, err := tensor.Div(rmsDelta, rmsGrad)
		if err != nil {
			return err
		}
		update, err := tensor.Mul(ratio, grad)
		if err != nil {
			return err
		}
		if o.lr != 1.0 {
			update.Scale(o.lr)
		}
		if err := p.AddScaled(update, -1); err != nil {
			return err
		}
		deltaSquared := update.Clone()
		if err := deltaSquared.MulInPlace(update); err != nil {
			return err
		}
		dAvg.Scale(o.rho)
		if err := dAvg.AddScaled(deltaSquared, 1-o.rho); err != nil {
			return err
		}
		o.squareAvg[p] = sqAvg
		o.deltaAvg[p] = dAvg
	}
	return nil
}

func (o *Adadelta) ZeroGrad() {
	for _, p := range o.params {
		if p != nil {
			p.ZeroGrad()
		}
	}
}
