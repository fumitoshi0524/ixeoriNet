package nn

import (
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

// ConvTranspose3d implements a 3D transposed convolution layer.
type ConvTranspose3d struct {
	weight *tensor.Tensor
	bias   *tensor.Tensor

	strideD int
	strideH int
	strideW int
	padD    int
	padH    int
	padW    int
}

// NewConvTranspose3d creates a ConvTranspose3d module.
// weight shape: [in_channels, out_channels, kernelD, kernelH, kernelW]
func NewConvTranspose3d(inChannels, outChannels, kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW int, withBias bool) *ConvTranspose3d {
	if strideD <= 0 {
		strideD = 1
	}
	if strideH <= 0 {
		strideH = 1
	}
	if strideW <= 0 {
		strideW = 1
	}
	weight := tensor.Randn(inChannels, outChannels, kernelD, kernelH, kernelW)
	fanIn := float64(inChannels * kernelD * kernelH * kernelW)
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

	return &ConvTranspose3d{
		weight:  weight,
		bias:    bias,
		strideD: strideD,
		strideH: strideH,
		strideW: strideW,
		padD:    padD,
		padH:    padH,
		padW:    padW,
	}
}

func (c *ConvTranspose3d) Parameters() []*tensor.Tensor {
	if c.bias == nil {
		return []*tensor.Tensor{c.weight}
	}
	return []*tensor.Tensor{c.weight, c.bias}
}

func (c *ConvTranspose3d) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.ConvTranspose3D(x, c.weight, c.bias, c.strideD, c.strideH, c.strideW, c.padD, c.padH, c.padW)
}

func (c *ConvTranspose3d) ZeroGrad() {
	c.weight.ZeroGrad()
	if c.bias != nil {
		c.bias.ZeroGrad()
	}
}

func (c *ConvTranspose3d) Train() {}

func (c *ConvTranspose3d) Eval() {}

func (c *ConvTranspose3d) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight")] = c.weight.Clone()
	if c.bias != nil {
		state[joinPrefix(prefix, "bias")] = c.bias.Clone()
	}
}

func (c *ConvTranspose3d) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	weightKey := joinPrefix(prefix, "weight")
	w, ok := state[weightKey]
	if !ok {
		return fmt.Errorf("ConvTranspose3d missing %s", weightKey)
	}
	if err := tensor.CopyInto(c.weight, w); err != nil {
		return fmt.Errorf("load %s: %w", weightKey, err)
	}
	if c.bias != nil {
		biasKey := joinPrefix(prefix, "bias")
		b, ok := state[biasKey]
		if !ok {
			return fmt.Errorf("ConvTranspose3d missing %s", biasKey)
		}
		if err := tensor.CopyInto(c.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", biasKey, err)
		}
	}
	return nil
}
