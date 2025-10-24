package nn

import "github.com/fumitoshi0524/ixeoriNet/tensor"

type Dropout struct {
	p        float64
	training bool
}

func NewDropout(p float64) *Dropout {
	if p < 0 {
		p = 0
	}
	if p >= 1 {
		p = 0.999 // clamp to avoid invalid probability
	}
	return &Dropout{p: p, training: true}
}

func (d *Dropout) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Dropout(input, d.p, d.training)
}

func (d *Dropout) Parameters() []*tensor.Tensor {
	return nil
}

func (d *Dropout) ZeroGrad() {}

func (d *Dropout) Train() {
	d.training = true
}

func (d *Dropout) Eval() {
	d.training = false
}
