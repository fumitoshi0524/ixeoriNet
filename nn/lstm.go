package nn

import (
	"errors"
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

const (
	lstmGateInput = iota
	lstmGateForget
	lstmGateCell
	lstmGateOutput
	lstmGateTotal
)

type LSTM struct {
	inputSize  int
	hiddenSize int
	withBias   bool

	weightIH [lstmGateTotal]*tensor.Tensor
	weightHH [lstmGateTotal]*tensor.Tensor
	biasIH   [lstmGateTotal]*tensor.Tensor
	biasHH   [lstmGateTotal]*tensor.Tensor
}

func NewLSTM(inputSize, hiddenSize int, withBias bool) *LSTM {
	l := &LSTM{
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		withBias:   withBias,
	}
	for gate := 0; gate < lstmGateTotal; gate++ {
		wIn := tensor.Randn(hiddenSize, inputSize)
		wHidden := tensor.Randn(hiddenSize, hiddenSize)
		inScale := math.Sqrt(1.0 / float64(inputSize))
		hidScale := math.Sqrt(1.0 / float64(hiddenSize))
		wIn.Scale(inScale)
		wHidden.Scale(hidScale)
		wIn.SetRequiresGrad(true)
		wHidden.SetRequiresGrad(true)
		l.weightIH[gate] = wIn
		l.weightHH[gate] = wHidden
		if withBias {
			bIn := tensor.Zeros(hiddenSize)
			bHidden := tensor.Zeros(hiddenSize)
			bIn.SetRequiresGrad(true)
			bHidden.SetRequiresGrad(true)
			l.biasIH[gate] = bIn
			l.biasHH[gate] = bHidden
		}
	}
	return l
}

func (l *LSTM) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	out, _, _, err := l.ForwardWithState(input, nil, nil)
	return out, err
}

func (l *LSTM) ForwardWithState(input *tensor.Tensor, hx, cx *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	shape := input.Shape()
	if len(shape) != 3 {
		return nil, nil, nil, errors.New("LSTM expects input shape [seq, batch, features]")
	}
	seqLen := shape[0]
	batch := shape[1]
	features := shape[2]
	if features != l.inputSize {
		return nil, nil, nil, errors.New("input feature mismatch")
	}
	if hx == nil {
		hx = tensor.Zeros(batch, l.hiddenSize)
	}
	if cx == nil {
		cx = tensor.Zeros(batch, l.hiddenSize)
	}
	hShape := hx.Shape()
	cShape := cx.Shape()
	if len(hShape) != 2 || hShape[0] != batch || hShape[1] != l.hiddenSize {
		return nil, nil, nil, errors.New("hidden state shape mismatch")
	}
	if len(cShape) != 2 || cShape[0] != batch || cShape[1] != l.hiddenSize {
		return nil, nil, nil, errors.New("cell state shape mismatch")
	}

	sizes := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		sizes[i] = 1
	}
	steps, err := tensor.Split(0, sizes, input)
	if err != nil {
		return nil, nil, nil, err
	}
	currentH := hx
	currentC := cx
	frames := make([]*tensor.Tensor, 0, seqLen)
	for _, step := range steps {
		var x *tensor.Tensor
		x, err = step.Reshape(batch, features)
		if err != nil {
			return nil, nil, nil, err
		}

		iPre, err := l.affine(x, currentH, lstmGateInput)
		if err != nil {
			return nil, nil, nil, err
		}
		inputGate := tensor.Sigmoid(iPre)

		fPre, err := l.affine(x, currentH, lstmGateForget)
		if err != nil {
			return nil, nil, nil, err
		}
		forgetGate := tensor.Sigmoid(fPre)

		gPre, err := l.affine(x, currentH, lstmGateCell)
		if err != nil {
			return nil, nil, nil, err
		}
		cellCandidate := tensor.Tanh(gPre)

		oPre, err := l.affine(x, currentH, lstmGateOutput)
		if err != nil {
			return nil, nil, nil, err
		}
		outputGate := tensor.Sigmoid(oPre)

		t1, err := tensor.Mul(forgetGate, currentC)
		if err != nil {
			return nil, nil, nil, err
		}
		t2, err := tensor.Mul(inputGate, cellCandidate)
		if err != nil {
			return nil, nil, nil, err
		}
		currentC, err = tensor.Add(t1, t2)
		if err != nil {
			return nil, nil, nil, err
		}

		tanhC := tensor.Tanh(currentC)
		currentH, err = tensor.Mul(outputGate, tanhC)
		if err != nil {
			return nil, nil, nil, err
		}

		frame, err := tensor.Unsqueeze(currentH, 0)
		if err != nil {
			return nil, nil, nil, err
		}
		frames = append(frames, frame)
	}

	output, err := tensor.Concat(0, frames...)
	if err != nil {
		return nil, nil, nil, err
	}
	return output, currentH, currentC, nil
}

func (l *LSTM) affine(x, h *tensor.Tensor, gate int) (*tensor.Tensor, error) {
	inputPart, err := tensor.MatMul(x, l.weightIH[gate].MustTranspose())
	if err != nil {
		return nil, err
	}
	hiddenPart, err := tensor.MatMul(h, l.weightHH[gate].MustTranspose())
	if err != nil {
		return nil, err
	}
	sum, err := tensor.Add(inputPart, hiddenPart)
	if err != nil {
		return nil, err
	}
	if l.withBias {
		sum, err = tensor.AddBias2D(sum, l.biasIH[gate])
		if err != nil {
			return nil, err
		}
		sum, err = tensor.AddBias2D(sum, l.biasHH[gate])
		if err != nil {
			return nil, err
		}
	}
	return sum, nil
}

func (l *LSTM) Parameters() []*tensor.Tensor {
	params := make([]*tensor.Tensor, 0, lstmGateTotal*4)
	for gate := 0; gate < lstmGateTotal; gate++ {
		params = append(params, l.weightIH[gate], l.weightHH[gate])
		if l.withBias {
			params = append(params, l.biasIH[gate], l.biasHH[gate])
		}
	}
	return params
}

func (l *LSTM) ZeroGrad() {
	for _, p := range l.Parameters() {
		if p != nil {
			p.ZeroGrad()
		}
	}
}

func (l *LSTM) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	gateNames := []string{"input", "forget", "cell", "output"}
	for gate, name := range gateNames {
		state[joinPrefix(prefix, "weight_ih_"+name)] = l.weightIH[gate].Clone()
		state[joinPrefix(prefix, "weight_hh_"+name)] = l.weightHH[gate].Clone()
		if l.withBias {
			state[joinPrefix(prefix, "bias_ih_"+name)] = l.biasIH[gate].Clone()
			state[joinPrefix(prefix, "bias_hh_"+name)] = l.biasHH[gate].Clone()
		}
	}
}

func (l *LSTM) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	gateNames := []string{"input", "forget", "cell", "output"}
	for gate, name := range gateNames {
		wIHKey := joinPrefix(prefix, "weight_ih_"+name)
		wIH, ok := state[wIHKey]
		if !ok {
			return fmt.Errorf("LSTM missing %s", wIHKey)
		}
		if err := tensor.CopyInto(l.weightIH[gate], wIH); err != nil {
			return fmt.Errorf("load %s: %w", wIHKey, err)
		}
		wHHKey := joinPrefix(prefix, "weight_hh_"+name)
		wHH, ok := state[wHHKey]
		if !ok {
			return fmt.Errorf("LSTM missing %s", wHHKey)
		}
		if err := tensor.CopyInto(l.weightHH[gate], wHH); err != nil {
			return fmt.Errorf("load %s: %w", wHHKey, err)
		}
		if l.withBias {
			bIHKey := joinPrefix(prefix, "bias_ih_"+name)
			bIH, ok := state[bIHKey]
			if !ok {
				return fmt.Errorf("LSTM missing %s", bIHKey)
			}
			if err := tensor.CopyInto(l.biasIH[gate], bIH); err != nil {
				return fmt.Errorf("load %s: %w", bIHKey, err)
			}
			bHHKey := joinPrefix(prefix, "bias_hh_"+name)
			bHH, ok := state[bHHKey]
			if !ok {
				return fmt.Errorf("LSTM missing %s", bHHKey)
			}
			if err := tensor.CopyInto(l.biasHH[gate], bHH); err != nil {
				return fmt.Errorf("load %s: %w", bHHKey, err)
			}
		}
	}
	return nil
}
