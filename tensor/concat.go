package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func Concat(axis int, tensors ...*Tensor) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("Concat requires at least one tensor")
	}
	base := tensors[0]
	rank := len(base.shape)
	if rank == 0 {
		return nil, errors.New("Concat requires rank >= 1")
	}
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, errors.New("axis out of range")
	}
	outShape := append([]int(nil), base.shape...)
	sumAxis := base.shape[axis]
	for i := 1; i < len(tensors); i++ {
		t := tensors[i]
		if len(t.shape) != rank {
			return nil, errors.New("rank mismatch")
		}
		for d := 0; d < rank; d++ {
			if d == axis {
				continue
			}
			if t.shape[d] != base.shape[d] {
				return nil, errors.New("shape mismatch")
			}
		}
		sumAxis += t.shape[axis]
	}
	outShape[axis] = sumAxis
	size := 1
	for _, dim := range outShape {
		size *= dim
	}
	out := MustNew(make([]float64, size), outShape...)
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= base.shape[i]
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= base.shape[i]
	}
	axisOffset := 0
	for _, t := range tensors {
		axisSize := t.shape[axis]
		parallel.For(outer, func(start, end int) {
			for o := start; o < end; o++ {
				dstStart := (o*outShape[axis] + axisOffset) * inner
				srcStart := o * axisSize * inner
				copy(out.data[dstStart:dstStart+axisSize*inner], t.data[srcStart:srcStart+axisSize*inner])
			}
		})
		axisOffset += axisSize
	}
	reqGrad := false
	for _, t := range tensors {
		if t.requiresGrad {
			reqGrad = true
			break
		}
	}
	if reqGrad {
		out.requiresGrad = true
		parents := make([]*Tensor, 0, len(tensors))
		for _, t := range tensors {
			if t.requiresGrad {
				parents = append(parents, t)
			}
		}
		out.parents = parents
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				offset := 0
				for _, t := range tensors {
					axisSize := t.shape[axis]
					if t.requiresGrad {
						g := Zeros(t.shape...)
						parallel.For(outer, func(start, end int) {
							for o := start; o < end; o++ {
								srcStart := (o*outShape[axis] + offset) * inner
								dstStart := o * axisSize * inner
								copy(g.data[dstStart:dstStart+axisSize*inner], grad.data[srcStart:srcStart+axisSize*inner])
							}
						})
						accumulate(grads, t, g)
					}
					offset += axisSize
				}
			},
		}
	}
	return out, nil
}
