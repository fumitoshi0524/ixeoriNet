package nn

import (
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Conv2d struct {
	inChannels  int
	outChannels int
	kernelH     int
	kernelW     int
	strideH     int
	strideW     int
	padH        int
	padW        int
	weight      *tensor.Tensor
	bias        *tensor.Tensor
}

func NewConv2d(inChannels, outChannels, kernelH, kernelW int, strideH, strideW, padH, padW int, withBias bool) *Conv2d {
	if strideH <= 0 {
		strideH = 1
	}
	if strideW <= 0 {
		strideW = 1
	}
	w := tensor.Randn(outChannels, inChannels, kernelH, kernelW)
	fanIn := float64(inChannels * kernelH * kernelW)
	scale := math.Sqrt(2.0 / fanIn)
	w.Scale(scale)
	w.SetRequiresGrad(true)
	var b *tensor.Tensor
	if withBias {
		b = tensor.Zeros(outChannels)
		b.SetRequiresGrad(true)
	}
	return &Conv2d{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelH:     kernelH,
		kernelW:     kernelW,
		strideH:     strideH,
		strideW:     strideW,
		padH:        padH,
		padW:        padW,
		weight:      w,
		bias:        b,
	}
}

func (c *Conv2d) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Conv2D(input, c.weight, c.bias, c.strideH, c.strideW, c.padH, c.padW)
}

func (c *Conv2d) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{c.weight}
	if c.bias != nil {
		params = append(params, c.bias)
	}
	return params
}

func (c *Conv2d) ZeroGrad() {
	for _, p := range c.Parameters() {
		p.ZeroGrad()
	}
}

func (c *Conv2d) Weight() *tensor.Tensor {
	return c.weight
}

func (c *Conv2d) Bias() *tensor.Tensor {
	return c.bias
}

func (c *Conv2d) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight")] = c.weight.Clone()
	if c.bias != nil {
		state[joinPrefix(prefix, "bias")] = c.bias.Clone()
	}
}

func (c *Conv2d) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	weightKey := joinPrefix(prefix, "weight")
	w, ok := state[weightKey]
	if !ok {
		return fmt.Errorf("Conv2d missing %s", weightKey)
	}
	if err := tensor.CopyInto(c.weight, w); err != nil {
		return fmt.Errorf("load %s: %w", weightKey, err)
	}
	if c.bias != nil {
		biasKey := joinPrefix(prefix, "bias")
		b, ok := state[biasKey]
		if !ok {
			return fmt.Errorf("Conv2d missing %s", biasKey)
		}
		if err := tensor.CopyInto(c.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", biasKey, err)
		}
	}
	return nil
}
