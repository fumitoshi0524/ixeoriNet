package nn

import (
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Conv3d struct {
	inChannels  int
	outChannels int
	kernelD     int
	kernelH     int
	kernelW     int
	strideD     int
	strideH     int
	strideW     int
	padD        int
	padH        int
	padW        int
	weight      *tensor.Tensor
	bias        *tensor.Tensor
}

// NewConv3d constructs a Conv3d module.
func NewConv3d(inChannels, outChannels, kernelD, kernelH, kernelW int, strideD, strideH, strideW, padD, padH, padW int, withBias bool) *Conv3d {
	if strideD <= 0 {
		strideD = 1
	}
	if strideH <= 0 {
		strideH = 1
	}
	if strideW <= 0 {
		strideW = 1
	}
	weight := tensor.Randn(outChannels, inChannels, kernelD, kernelH, kernelW)
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

	return &Conv3d{
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelD:     kernelD,
		kernelH:     kernelH,
		kernelW:     kernelW,
		strideD:     strideD,
		strideH:     strideH,
		strideW:     strideW,
		padD:        padD,
		padH:        padH,
		padW:        padW,
		weight:      weight,
		bias:        bias,
	}
}

func (c *Conv3d) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Conv3D(input, c.weight, c.bias, c.strideD, c.strideH, c.strideW, c.padD, c.padH, c.padW)
}

func (c *Conv3d) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{c.weight}
	if c.bias != nil {
		params = append(params, c.bias)
	}
	return params
}

func (c *Conv3d) ZeroGrad() {
	for _, p := range c.Parameters() {
		p.ZeroGrad()
	}
}

func (c *Conv3d) Train() {}

func (c *Conv3d) Eval() {}

func (c *Conv3d) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight")] = c.weight.Clone()
	if c.bias != nil {
		state[joinPrefix(prefix, "bias")] = c.bias.Clone()
	}
}

func (c *Conv3d) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	weightKey := joinPrefix(prefix, "weight")
	w, ok := state[weightKey]
	if !ok {
		return fmt.Errorf("Conv3d missing %s", weightKey)
	}
	if err := tensor.CopyInto(c.weight, w); err != nil {
		return fmt.Errorf("load %s: %w", weightKey, err)
	}
	if c.bias != nil {
		biasKey := joinPrefix(prefix, "bias")
		b, ok := state[biasKey]
		if !ok {
			return fmt.Errorf("Conv3d missing %s", biasKey)
		}
		if err := tensor.CopyInto(c.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", biasKey, err)
		}
	}
	return nil
}
