package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func Split(axis int, sizes []int, t *Tensor) ([]*Tensor, error) {
	if len(sizes) == 0 {
		return nil, errors.New("Split requires at least one size")
	}
	rank := len(t.shape)
	if rank == 0 {
		return nil, errors.New("Split requires rank >= 1 tensor")
	}
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, errors.New("axis out of range")
	}
	total := 0
	for _, s := range sizes {
		if s <= 0 {
			return nil, errors.New("split sizes must be positive")
		}
		total += s
	}
	if total != t.shape[axis] {
		return nil, errors.New("split sizes do not match tensor axis length")
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= t.shape[i]
	}
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= t.shape[i]
	}
	result := make([]*Tensor, len(sizes))
	offset := 0
	for idx, size := range sizes {
		shape := append([]int(nil), t.shape...)
		shape[axis] = size
		data := make([]float64, size*outer*inner)
		parallel.For(outer, func(start, end int) {
			for o := start; o < end; o++ {
				src := (o*t.shape[axis] + offset) * inner
				dst := o * size * inner
				copy(data[dst:dst+size*inner], t.data[src:src+size*inner])
			}
		})
		result[idx] = MustNew(data, shape...)
		offset += size
	}
	requires := t.requiresGrad
	if requires {
		offset := 0
		for i, part := range result {
			part.requiresGrad = true
			part.parents = []*Tensor{t}
			size := sizes[i]
			partOffset := offset
			part.node = &node{
				backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
					gFull := Zeros(t.shape...)
					parallel.For(outer, func(start, end int) {
						for o := start; o < end; o++ {
							dst := (o*t.shape[axis] + partOffset) * inner
							src := o * size * inner
							copy(gFull.data[dst:dst+size*inner], grad.data[src:src+size*inner])
						}
					})
					accumulate(grads, t, gFull)
				},
			}
			offset += size
		}
	}
	return result, nil
}

func Chunk(axis int, parts int, t *Tensor) ([]*Tensor, error) {
	if parts <= 0 {
		return nil, errors.New("parts must be positive")
	}
	rank := len(t.shape)
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, errors.New("axis out of range")
	}
	size := t.shape[axis]
	base := size / parts
	remainder := size % parts
	sizes := make([]int, 0, parts)
	for i := 0; i < parts; i++ {
		chunk := base
		if i < remainder {
			chunk++
		}
		sizes = append(sizes, chunk)
	}
	return Split(axis, sizes, t)
}
