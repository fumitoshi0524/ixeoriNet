package tensor

import "errors"

func (t *Tensor) Reshape(shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, errors.New("reshape shape required")
	}
	total := t.Numel()
	prod := 1
	infer := -1
	for i, dim := range shape {
		if dim == -1 {
			if infer != -1 {
				return nil, errors.New("multiple inferred dimensions")
			}
			infer = i
			continue
		}
		if dim <= 0 {
			return nil, errors.New("invalid reshape dimension")
		}
		prod *= dim
	}
	if infer != -1 {
		if prod == 0 || total%prod != 0 {
			return nil, errors.New("cannot infer dimension")
		}
		shape[infer] = total / prod
		prod = total
	}
	if prod != total {
		return nil, errors.New("reshape size mismatch")
	}
	out := &Tensor{
		data:         t.data,
		shape:        append([]int(nil), shape...),
		strides:      makeStrides(shape),
		requiresGrad: t.requiresGrad,
	}
	if t.requiresGrad {
		out.parents = []*Tensor{t}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				reshaped := grad.Clone()
				reshaped.shape = append([]int(nil), t.shape...)
				reshaped.strides = makeStrides(reshaped.shape)
				accumulate(grads, t, reshaped)
			},
		}
	}
	return out, nil
}

func Flatten(a *Tensor) (*Tensor, error) {
	if len(a.shape) < 2 {
		return a.Reshape(a.Numel())
	}
	batch := a.shape[0]
	features := 1
	for _, dim := range a.shape[1:] {
		features *= dim
	}
	return a.Reshape(batch, features)
}
