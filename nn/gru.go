package nn

import (
	"errors"
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

const (
	gruGateUpdate = iota
	gruGateReset
	gruGateNew
	gruGateTotal
)

type GRU struct {
	inputSize  int
	hiddenSize int
	withBias   bool

	weightIH [gruGateTotal]*tensor.Tensor
	weightHH [gruGateTotal]*tensor.Tensor
	biasIH   [gruGateTotal]*tensor.Tensor
	biasHH   [gruGateTotal]*tensor.Tensor
}

func NewGRU(inputSize, hiddenSize int, withBias bool) *GRU {
	g := &GRU{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		withBias:   withBias,
	}
	for gate := 0; gate < gruGateTotal; gate++ {
		wIn := tensor.Randn(hiddenSize, inputSize)
		wHidden := tensor.Randn(hiddenSize, hiddenSize)
		inScale := math.Sqrt(1.0 / float64(inputSize))
		hidScale := math.Sqrt(1.0 / float64(hiddenSize))
		wIn.Scale(inScale)
		wHidden.Scale(hidScale)
		wIn.SetRequiresGrad(true)
		wHidden.SetRequiresGrad(true)
		g.weightIH[gate] = wIn
		g.weightHH[gate] = wHidden
		if withBias {
			bIn := tensor.Zeros(hiddenSize)
			bHidden := tensor.Zeros(hiddenSize)
			bIn.SetRequiresGrad(true)
			bHidden.SetRequiresGrad(true)
			g.biasIH[gate] = bIn
			g.biasHH[gate] = bHidden
		}
	}
	return g
}

func (g *GRU) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	out, _, err := g.ForwardWithState(input, nil)
	return out, err
}

func (g *GRU) ForwardWithState(input *tensor.Tensor, hx *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, nil, errors.New("GRU expects input shape [seq, batch, features]")
	}
	seqLen := shape[0]
	batch := shape[1]
	features := shape[2]
	if features != g.inputSize {
		return nil, nil, errors.New("input feature mismatch")
	}
	if hx == nil {
		hx = tensor.Zeros(batch, g.hiddenSize)
	}
	hShape := hx.Shape()
	if len(hShape) != 2 || hShape[0] != batch || hShape[1] != g.hiddenSize {
		return nil, nil, errors.New("hidden state shape mismatch")
	}

	sizes := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		sizes[i] = 1
	}
	steps, err := tensor.Split(0, sizes, input)
	if err != nil {
		return nil, nil, err
	}
	current := hx
	frames := make([]*tensor.Tensor, 0, seqLen)
	for _, step := range steps {
		var x *tensor.Tensor
		x, err = step.Reshape(batch, features)
		if err != nil {
			return nil, nil, err
		}

		zPre, err := g.affine(x, current, gruGateUpdate)
		if err != nil {
			return nil, nil, err
		}
		z := tensor.Sigmoid(zPre)

		rPre, err := g.affine(x, current, gruGateReset)
		if err != nil {
			return nil, nil, err
		}
		r := tensor.Sigmoid(rPre)

		rHidden, err := tensor.Mul(r, current)
		if err != nil {
			return nil, nil, err
		}
		nPreInput, err := tensor.MatMul(x, g.weightIH[gruGateNew].MustTranspose())
		if err != nil {
			return nil, nil, err
		}
		nPreHidden, err := tensor.MatMul(rHidden, g.weightHH[gruGateNew].MustTranspose())
		if err != nil {
			return nil, nil, err
		}
		nPre, err := tensor.Add(nPreInput, nPreHidden)
		if err != nil {
			return nil, nil, err
		}
		if g.withBias {
			nPre, err = tensor.AddBias2D(nPre, g.biasIH[gruGateNew])
			if err != nil {
				return nil, nil, err
			}
			nPre, err = tensor.AddBias2D(nPre, g.biasHH[gruGateNew])
			if err != nil {
				return nil, nil, err
			}
		}
		nCandidate := tensor.Tanh(nPre)

		shapeHidden := z.Shape()
		ones := tensor.Ones(shapeHidden...)
		oneMinusZ, err := tensor.Sub(ones, z)
		if err != nil {
			return nil, nil, err
		}
		part1, err := tensor.Mul(oneMinusZ, nCandidate)
		if err != nil {
			return nil, nil, err
		}
		part2, err := tensor.Mul(z, current)
		if err != nil {
			return nil, nil, err
		}
		current, err = tensor.Add(part1, part2)
		if err != nil {
			return nil, nil, err
		}

		frame, err := tensor.Unsqueeze(current, 0)
		if err != nil {
			return nil, nil, err
		}
		frames = append(frames, frame)
	}

	output, err := tensor.Concat(0, frames...)
	if err != nil {
		return nil, nil, err
	}
	return output, current, nil
}

func (g *GRU) affine(x, h *tensor.Tensor, gate int) (*tensor.Tensor, error) {
	inputPart, err := tensor.MatMul(x, g.weightIH[gate].MustTranspose())
	if err != nil {
		return nil, err
	}
	hiddenPart, err := tensor.MatMul(h, g.weightHH[gate].MustTranspose())
	if err != nil {
		return nil, err
	}
	sum, err := tensor.Add(inputPart, hiddenPart)
	if err != nil {
		return nil, err
	}
	if g.withBias {
		sum, err = tensor.AddBias2D(sum, g.biasIH[gate])
		if err != nil {
			return nil, err
		}
		sum, err = tensor.AddBias2D(sum, g.biasHH[gate])
		if err != nil {
			return nil, err
		}
	}
	return sum, nil
}

func (g *GRU) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0, gruGateTotal*4)
	for gate := 0; gate < gruGateTotal; gate++ {
		params = append(params, g.weightIH[gate], g.weightHH[gate])
		if g.withBias {
			params = append(params, g.biasIH[gate], g.biasHH[gate])
		}
	}
	return params
}

func (g *GRU) ZeroGrad() {
	for _, p := range g.Parameters() {
		if p != nil {
			p.ZeroGrad()
		}
	}
}

func (g *GRU) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	gateNames := []string{"update", "reset", "new"}
	for gate, name := range gateNames {
		state[joinPrefix(prefix, "weight_ih_"+name)] = g.weightIH[gate].Clone()
		state[joinPrefix(prefix, "weight_hh_"+name)] = g.weightHH[gate].Clone()
		if g.withBias {
			state[joinPrefix(prefix, "bias_ih_"+name)] = g.biasIH[gate].Clone()
			state[joinPrefix(prefix, "bias_hh_"+name)] = g.biasHH[gate].Clone()
		}
	}
}

func (g *GRU) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	gateNames := []string{"update", "reset", "new"}
	for gate, name := range gateNames {
		wIHKey := joinPrefix(prefix, "weight_ih_"+name)
		wIH, ok := state[wIHKey]
		if !ok {
			return fmt.Errorf("GRU missing %s", wIHKey)
		}
		if err := tensor.CopyInto(g.weightIH[gate], wIH); err != nil {
			return fmt.Errorf("load %s: %w", wIHKey, err)
		}
		wHHKey := joinPrefix(prefix, "weight_hh_"+name)
		wHH, ok := state[wHHKey]
		if !ok {
			return fmt.Errorf("GRU missing %s", wHHKey)
		}
		if err := tensor.CopyInto(g.weightHH[gate], wHH); err != nil {
			return fmt.Errorf("load %s: %w", wHHKey, err)
		}
		if g.withBias {
			bIHKey := joinPrefix(prefix, "bias_ih_"+name)
			bIH, ok := state[bIHKey]
			if !ok {
				return fmt.Errorf("GRU missing %s", bIHKey)
			}
			if err := tensor.CopyInto(g.biasIH[gate], bIH); err != nil {
				return fmt.Errorf("load %s: %w", bIHKey, err)
			}
			bHHKey := joinPrefix(prefix, "bias_hh_"+name)
			bHH, ok := state[bHHKey]
			if !ok {
				return fmt.Errorf("GRU missing %s", bHHKey)
			}
			if err := tensor.CopyInto(g.biasHH[gate], bHH); err != nil {
				return fmt.Errorf("load %s: %w", bHHKey, err)
			}
		}
	}
	return nil
}
