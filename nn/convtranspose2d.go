package nn

import (
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

// ConvTranspose2d implements a 2D transposed convolution module.
type ConvTranspose2d struct {
	weight *tensor.Tensor
	bias   *tensor.Tensor

	strideH int
	strideW int
	padH    int
	padW    int
}

// NewConvTranspose2d creates a ConvTranspose2d module.
// weight shape: [in_channels, out_channels, kernelH, kernelW]
func NewConvTranspose2d(inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int, withBias bool) *ConvTranspose2d {
	if strideH <= 0 {
		strideH = 1
	}
	if strideW <= 0 {
		strideW = 1
	}
	weight := tensor.Randn(inChannels, outChannels, kernelH, kernelW)
	fanIn := float64(inChannels * kernelH * kernelW)
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

	return &ConvTranspose2d{
		weight:  weight,
		bias:    bias,
		strideH: strideH,
		strideW: strideW,
		padH:    padH,
		padW:    padW,
	}
}

func (c *ConvTranspose2d) Parameters() []*tensor.Tensor {
	if c.bias == nil {
		return []*tensor.Tensor{c.weight}
	}
	return []*tensor.Tensor{c.weight, c.bias}
}

func (c *ConvTranspose2d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.ConvTranspose2D(x, c.weight, c.bias, c.strideH, c.strideW, c.padH, c.padW)
}

func (c *ConvTranspose2d) ZeroGrad() {
	c.weight.ZeroGrad()
	if c.bias != nil {
		c.bias.ZeroGrad()
	}
}

func (c *ConvTranspose2d) Train() {}

func (c *ConvTranspose2d) Eval() {}

func (c *ConvTranspose2d) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight")] = c.weight.Clone()
	if c.bias != nil {
		state[joinPrefix(prefix, "bias")] = c.bias.Clone()
	}
}

func (c *ConvTranspose2d) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	weightKey := joinPrefix(prefix, "weight")
	w, ok := state[weightKey]
	if !ok {
		return fmt.Errorf("ConvTranspose2d missing %s", weightKey)
	}
	if err := tensor.CopyInto(c.weight, w); err != nil {
		return fmt.Errorf("load %s: %w", weightKey, err)
	}
	if c.bias != nil {
		biasKey := joinPrefix(prefix, "bias")
		b, ok := state[biasKey]
		if !ok {
			return fmt.Errorf("ConvTranspose2d missing %s", biasKey)
		}
		if err := tensor.CopyInto(c.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", biasKey, err)
		}
	}
	return nil
}
