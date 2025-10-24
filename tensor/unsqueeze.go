package tensor

import "errors"

func Unsqueeze(t *Tensor, axis int) (*Tensor, error) {
	rank := len(t.shape)
	if axis < 0 {
		axis += rank + 1
	}
	if axis < 0 || axis > rank {
		return nil, errors.New("axis out of range")
	}
	newShape := make([]int, rank+1)
	copy(newShape[:axis], t.shape[:axis])
	newShape[axis] = 1
	copy(newShape[axis+1:], t.shape[axis:])
	out := &Tensor{
		data:         t.data,
		shape:        newShape,
		strides:      makeStrides(newShape),
		requiresGrad: t.requiresGrad,
	}
	if t.requiresGrad {
		out.parents = []*Tensor{t}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				reshaped := &Tensor{
					data:    grad.data,
					shape:   append([]int(nil), t.shape...),
					strides: makeStrides(t.shape),
				}
				accumulate(grads, t, reshaped)
			},
		}
	}
	return out, nil
}
