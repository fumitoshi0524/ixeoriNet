package nn

import (
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Linear struct {
	inFeatures  int
	outFeatures int
	weight      *tensor.Tensor
	bias        *tensor.Tensor
}

func NewLinear(inFeatures, outFeatures int, withBias bool) *Linear {
	w := tensor.Randn(outFeatures, inFeatures)
	scale := math.Sqrt(2.0 / float64(inFeatures+outFeatures))
	w.Scale(scale)
	w.SetRequiresGrad(true)
	var b *tensor.Tensor
	if withBias {
		b = tensor.Randn(outFeatures)
		b.Scale(scale)
		b.SetRequiresGrad(true)
	}
	return &Linear{inFeatures: inFeatures, outFeatures: outFeatures, weight: w, bias: b}
}

func (l *Linear) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	shape := input.Shape()
	x := input
	var err error
	switch len(shape) {
	case 1:
		x, err = input.Reshape(1, shape[0])
	case 2:
		x = input
	default:
		x, err = tensor.Flatten(input)
	}
	if err != nil {
		return nil, err
	}
	wt := l.weight.MustTranspose()
	output, err := tensor.MatMul(x, wt)
	if err != nil {
		return nil, err
	}
	if l.bias != nil {
		output, err = tensor.AddBias2D(output, l.bias)
		if err != nil {
			return nil, err
		}
	}
	return output, nil
}

func (l *Linear) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{l.weight}
	if l.bias != nil {
		params = append(params, l.bias)
	}
	return params
}

func (l *Linear) ZeroGrad() {
	for _, p := range l.Parameters() {
		p.ZeroGrad()
	}
}

func (l *Linear) Weight() *tensor.Tensor {
	return l.weight
}

func (l *Linear) Bias() *tensor.Tensor {
	return l.bias
}

func (l *Linear) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight")] = l.weight.Clone()
	if l.bias != nil {
		state[joinPrefix(prefix, "bias")] = l.bias.Clone()
	}
}

func (l *Linear) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	weightKey := joinPrefix(prefix, "weight")
	w, ok := state[weightKey]
	if !ok {
		return fmt.Errorf("Linear missing %s", weightKey)
	}
	if err := tensor.CopyInto(l.weight, w); err != nil {
		return fmt.Errorf("load %s: %w", weightKey, err)
	}
	if l.bias != nil {
		biasKey := joinPrefix(prefix, "bias")
		b, ok := state[biasKey]
		if !ok {
			return fmt.Errorf("Linear missing %s", biasKey)
		}
		if err := tensor.CopyInto(l.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", biasKey, err)
		}
	}
	return nil
}
