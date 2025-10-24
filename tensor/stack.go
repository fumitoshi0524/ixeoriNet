package tensor

import "errors"

func Stack(axis int, tensors ...*Tensor) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("Stack requires at least one tensor")
	}
	base := tensors[0]
	rank := len(base.shape)
	for _, t := range tensors[1:] {
		if len(t.shape) != rank {
			return nil, errors.New("rank mismatch")
		}
		for i := range t.shape {
			if t.shape[i] != base.shape[i] {
				return nil, errors.New("shape mismatch")
			}
		}
	}
	if axis < 0 {
		axis += rank + 1
	}
	if axis < 0 || axis > rank {
		return nil, errors.New("axis out of range")
	}
	unsqueezed := make([]*Tensor, len(tensors))
	for i, t := range tensors {
		u, err := Unsqueeze(t, axis)
		if err != nil {
			return nil, err
		}
		unsqueezed[i] = u
	}
	return Concat(axis, unsqueezed...)
}
