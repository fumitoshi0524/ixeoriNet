package nn

import (
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Conv1d struct {
	inChannels  int
	outChannels int
	kernelW     int
	stride      int
	pad         int
	weight      *tensor.Tensor
	bias        *tensor.Tensor
}

func NewConv1d(inChannels, outChannels, kernelW, stride, pad int, withBias bool) *Conv1d {
	if stride <= 0 {
		stride = 1
	}
	w := tensor.Randn(outChannels, inChannels, kernelW)
	fanIn := float64(inChannels * kernelW)
	scale := math.Sqrt(2.0 / fanIn)
	w.Scale(scale)
	w.SetRequiresGrad(true)
	var b *tensor.Tensor
	if withBias {
		b = tensor.Zeros(outChannels)
		b.SetRequiresGrad(true)
	}
	return &Conv1d{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelW:     kernelW,
		stride:      stride,
		pad:         pad,
		weight:      w,
		bias:        b,
	}
}

func (c *Conv1d) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Conv1D(input, c.weight, c.bias, c.stride, c.pad)
}

func (c *Conv1d) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{c.weight}
	if c.bias != nil {
		params = append(params, c.bias)
	}
	return params
}

func (c *Conv1d) ZeroGrad() {
	for _, p := range c.Parameters() {
		p.ZeroGrad()
	}
}

func (c *Conv1d) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight")] = c.weight.Clone()
	if c.bias != nil {
		state[joinPrefix(prefix, "bias")] = c.bias.Clone()
	}
}

func (c *Conv1d) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	weightKey := joinPrefix(prefix, "weight")
	w, ok := state[weightKey]
	if !ok {
		return fmt.Errorf("Conv1d missing %s", weightKey)
	}
	if err := tensor.CopyInto(c.weight, w); err != nil {
		return fmt.Errorf("load %s: %w", weightKey, err)
	}
	if c.bias != nil {
		biasKey := joinPrefix(prefix, "bias")
		b, ok := state[biasKey]
		if !ok {
			return fmt.Errorf("Conv1d missing %s", biasKey)
		}
		if err := tensor.CopyInto(c.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", biasKey, err)
		}
	}
	return nil
}
