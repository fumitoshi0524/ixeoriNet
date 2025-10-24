package nn

import (
	"errors"
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type SimpleRNN struct {
	inputSize    int
	hiddenSize   int
	nonlinearity string
	weightIH     *tensor.Tensor
	weightHH     *tensor.Tensor
	biasIH       *tensor.Tensor
	biasHH       *tensor.Tensor
}

func NewSimpleRNN(inputSize, hiddenSize int, nonlinearity string, withBias bool) *SimpleRNN {
	if nonlinearity == "" {
		nonlinearity = "tanh"
	}
	weightIH := tensor.Randn(hiddenSize, inputSize)
	weightHH := tensor.Randn(hiddenSize, hiddenSize)
	scaleIH := math.Sqrt(1.0 / float64(inputSize))
	scaleHH := math.Sqrt(1.0 / float64(hiddenSize))
	weightIH.Scale(scaleIH)
	weightHH.Scale(scaleHH)
	weightIH.SetRequiresGrad(true)
	weightHH.SetRequiresGrad(true)
	var biasIH *tensor.Tensor
	var biasHH *tensor.Tensor
	if withBias {
		biasIH = tensor.Zeros(hiddenSize)
		biasHH = tensor.Zeros(hiddenSize)
		biasIH.SetRequiresGrad(true)
		biasHH.SetRequiresGrad(true)
	}
	return &SimpleRNN{
		inputSize:    inputSize,
		hiddenSize:   hiddenSize,
		nonlinearity: nonlinearity,
		weightIH:     weightIH,
		weightHH:     weightHH,
		biasIH:       biasIH,
		biasHH:       biasHH,
	}
}

func (r *SimpleRNN) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	out, _, err := r.ForwardWithState(input, nil)
	return out, err
}

func (r *SimpleRNN) ForwardWithState(input *tensor.Tensor, hx *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, nil, errors.New("SimpleRNN expects input shape [seq, batch, features]")
	}
	seqLen := shape[0]
	batch := shape[1]
	features := shape[2]
	if features != r.inputSize {
		return nil, nil, errors.New("input feature mismatch")
	}
	if hx == nil {
		hx = tensor.Zeros(batch, r.hiddenSize)
	}
	if len(hx.Shape()) != 2 || hx.Shape()[0] != batch || hx.Shape()[1] != r.hiddenSize {
		return nil, nil, errors.New("hidden state shape mismatch")
	}
	sizes := make([]int, seqLen)
	for i := range sizes {
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
		linear, err := tensor.MatMul(x, r.weightIH.MustTranspose())
		if err != nil {
			return nil, nil, err
		}
		hiddenLinear, err := tensor.MatMul(current, r.weightHH.MustTranspose())
		if err != nil {
			return nil, nil, err
		}
		summed, err := tensor.Add(linear, hiddenLinear)
		if err != nil {
			return nil, nil, err
		}
		if r.biasIH != nil {
			summed, err = tensor.AddBias2D(summed, r.biasIH)
			if err != nil {
				return nil, nil, err
			}
		}
		if r.biasHH != nil {
			summed, err = tensor.AddBias2D(summed, r.biasHH)
			if err != nil {
				return nil, nil, err
			}
		}
		current, err = r.activate(summed)
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

func (r *SimpleRNN) activate(t *tensor.Tensor) (*tensor.Tensor, error) {
	switch r.nonlinearity {
	case "relu":
		return tensor.Relu(t), nil
	case "tanh":
		return tensor.Tanh(t), nil
	default:
		return nil, errors.New("unsupported nonlinearity")
	}
}

func (r *SimpleRNN) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{r.weightIH, r.weightHH}
	if r.biasIH != nil {
		params = append(params, r.biasIH, r.biasHH)
	}
	return params
}

func (r *SimpleRNN) ZeroGrad() {
	for _, p := range r.Parameters() {
		p.ZeroGrad()
	}
}

func (r *SimpleRNN) WeightIH() *tensor.Tensor {
	return r.weightIH
}

func (r *SimpleRNN) WeightHH() *tensor.Tensor {
	return r.weightHH
}

func (r *SimpleRNN) BiasIH() *tensor.Tensor {
	return r.biasIH
}

func (r *SimpleRNN) BiasHH() *tensor.Tensor {
	return r.biasHH
}

func (r *SimpleRNN) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight_ih")] = r.weightIH.Clone()
	state[joinPrefix(prefix, "weight_hh")] = r.weightHH.Clone()
	if r.biasIH != nil {
		state[joinPrefix(prefix, "bias_ih")] = r.biasIH.Clone()
	}
	if r.biasHH != nil {
		state[joinPrefix(prefix, "bias_hh")] = r.biasHH.Clone()
	}
}

func (r *SimpleRNN) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	wIHKey := joinPrefix(prefix, "weight_ih")
	wIH, ok := state[wIHKey]
	if !ok {
		return fmt.Errorf("SimpleRNN missing %s", wIHKey)
	}
	if err := tensor.CopyInto(r.weightIH, wIH); err != nil {
		return fmt.Errorf("load %s: %w", wIHKey, err)
	}
	wHHKey := joinPrefix(prefix, "weight_hh")
	wHH, ok := state[wHHKey]
	if !ok {
		return fmt.Errorf("SimpleRNN missing %s", wHHKey)
	}
	if err := tensor.CopyInto(r.weightHH, wHH); err != nil {
		return fmt.Errorf("load %s: %w", wHHKey, err)
	}
	if r.biasIH != nil {
		bIHKey := joinPrefix(prefix, "bias_ih")
		bIH, ok := state[bIHKey]
		if !ok {
			return fmt.Errorf("SimpleRNN missing %s", bIHKey)
		}
		if err := tensor.CopyInto(r.biasIH, bIH); err != nil {
			return fmt.Errorf("load %s: %w", bIHKey, err)
		}
	}
	if r.biasHH != nil {
		bHHKey := joinPrefix(prefix, "bias_hh")
		bHH, ok := state[bHHKey]
		if !ok {
			return fmt.Errorf("SimpleRNN missing %s", bHHKey)
		}
		if err := tensor.CopyInto(r.biasHH, bHH); err != nil {
			return fmt.Errorf("load %s: %w", bHHKey, err)
		}
	}
	return nil
}
