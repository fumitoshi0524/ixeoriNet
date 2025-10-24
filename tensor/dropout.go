package tensor

import (
	"errors"
)

// Dropout applies dropout to the input tensor during training.
func Dropout(input *Tensor, p float64, training bool) (*Tensor, error) {
	if p < 0 || p >= 1 {
		return nil, errors.New("dropout probability must be in [0, 1)")
	}
	if !training || p == 0 {
		out := input.Clone()
		if input.requiresGrad {
			out.requiresGrad = true
			out.parents = []*Tensor{input}
			out.node = &node{
				backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
					accumulate(grads, input, grad)
				},
			}
		}
		return out, nil
	}

	scale := 1.0 / (1 - p)
	mask := make([]float64, len(input.data))
	out := Zeros(input.shape...)

	rngLock.Lock()
	for i := range mask {
		if rng.Float64() < p {
			mask[i] = 0
			out.data[i] = 0
		} else {
			mask[i] = scale
			out.data[i] = input.data[i] * scale
		}
	}
	rngLock.Unlock()

	if input.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{input}
		maskCopy := append([]float64(nil), mask...)
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				g := Zeros(input.shape...)
				for i := range g.data {
					g.data[i] = grad.data[i] * maskCopy[i]
				}
				accumulate(grads, input, g)
			},
		}
	}

	return out, nil
}
