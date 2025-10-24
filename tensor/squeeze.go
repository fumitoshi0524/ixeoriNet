package tensor

import (
	"errors"
	"sort"
)

func Squeeze(t *Tensor, axes ...int) (*Tensor, error) {
	originalShape := append([]int(nil), t.shape...)
	rank := len(originalShape)
	if rank == 0 {
		return nil, errors.New("Squeeze requires rank >= 1 tensor")
	}
	var toRemove []int
	if len(axes) == 0 {
		for i, dim := range originalShape {
			if dim == 1 {
				toRemove = append(toRemove, i)
			}
		}
	} else {
		seen := map[int]struct{}{}
		for _, axis := range axes {
			if axis < 0 {
				axis += rank
			}
			if axis < 0 || axis >= rank {
				return nil, errors.New("axis out of range")
			}
			if originalShape[axis] != 1 {
				return nil, errors.New("cannot squeeze axis with size > 1")
			}
			if _, ok := seen[axis]; !ok {
				seen[axis] = struct{}{}
				toRemove = append(toRemove, axis)
			}
		}
	}
	if len(toRemove) == 0 {
		out := t.Clone()
		out.requiresGrad = t.requiresGrad
		out.parents = nil
		out.node = nil
		return out, nil
	}
	sort.Ints(toRemove)
	newShape := make([]int, 0, rank-len(toRemove))
	next := 0
	for i := 0; i < rank; i++ {
		if next < len(toRemove) && toRemove[next] == i {
			next++
			continue
		}
		newShape = append(newShape, originalShape[i])
	}
	if len(newShape) == 0 {
		newShape = []int{1}
	}
	out := &Tensor{
		data:         t.data,
		shape:        append([]int(nil), newShape...),
		strides:      makeStrides(newShape),
		requiresGrad: t.requiresGrad,
	}
	if t.requiresGrad {
		out.parents = []*Tensor{t}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				shape := append([]int(nil), originalShape...)
				reshaped := &Tensor{
					data:    grad.data,
					shape:   shape,
					strides: makeStrides(shape),
				}
				accumulate(grads, t, reshaped)
			},
		}
	}
	return out, nil
}
