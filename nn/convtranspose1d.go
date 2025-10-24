package nn

import (
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

// ConvTranspose1d implements a 1D transposed convolution layer.
type ConvTranspose1d struct {
	weight *tensor.Tensor
	bias   *tensor.Tensor

	stride  int
	padding int
}

// NewConvTranspose1d creates a ConvTranspose1d module.
// weight shape: [in_channels, out_channels, kernel]
func NewConvTranspose1d(inChannels, outChannels, kernel, stride, padding int, withBias bool) *ConvTranspose1d {
	if stride <= 0 {
		stride = 1
	}
	weight := tensor.Randn(inChannels, outChannels, kernel)
	fanIn := float64(inChannels * kernel)
	if fanIn > 0 {
		scale := math.Sqrt(2.0 / fanIn)
		weight.Scale(scale)
	}
	weight.SetRequiresGrad(true)

	var bias *tensor.Tensor
	if withBias {
		bias = tensor.Zeros(outChannels)
		bias.SetRequiresGrad(true)
	}

	return &ConvTranspose1d{
		weight:  weight,
		bias:    bias,
		stride:  stride,
		padding: padding,
	}
}

func (c *ConvTranspose1d) Parameters() []*tensor.Tensor {
	if c.bias == nil {
		return []*tensor.Tensor{c.weight}
	}
	return []*tensor.Tensor{c.weight, c.bias}
}

func (c *ConvTranspose1d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.ConvTranspose1D(x, c.weight, c.bias, c.stride, c.padding)
}

func (c *ConvTranspose1d) ZeroGrad() {
	c.weight.ZeroGrad()
	if c.bias != nil {
		c.bias.ZeroGrad()
	}
}

func (c *ConvTranspose1d) Train() {}

func (c *ConvTranspose1d) Eval() {}

func (c *ConvTranspose1d) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight")] = c.weight.Clone()
	if c.bias != nil {
		state[joinPrefix(prefix, "bias")] = c.bias.Clone()
	}
}

func (c *ConvTranspose1d) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	weightKey := joinPrefix(prefix, "weight")
	w, ok := state[weightKey]
	if !ok {
		return fmt.Errorf("ConvTranspose1d missing %s", weightKey)
	}
	if err := tensor.CopyInto(c.weight, w); err != nil {
		return fmt.Errorf("load %s: %w", weightKey, err)
	}
	if c.bias != nil {
		biasKey := joinPrefix(prefix, "bias")
		b, ok := state[biasKey]
		if !ok {
			return fmt.Errorf("ConvTranspose1d missing %s", biasKey)
		}
		if err := tensor.CopyInto(c.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", biasKey, err)
		}
	}
	return nil
}
