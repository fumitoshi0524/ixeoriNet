package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func BroadcastTo(t *Tensor, targetShape []int) (*Tensor, error) {
	srcShape := t.shape
	srcRank := len(srcShape)
	tgtRank := len(targetShape)
	if tgtRank < srcRank {
		return nil, errors.New("target rank must be >= source rank")
	}
	off := tgtRank - srcRank
	newShape := append([]int(nil), targetShape...)
	strides := make([]int, tgtRank)
	for i := tgtRank - 1; i >= 0; i-- {
		var srcDim int
		var stride int
		if i-off >= 0 {
			srcDim = srcShape[i-off]
			stride = t.strides[i-off]
		} else {
			srcDim = 1
			stride = 0
		}
		tgtDim := targetShape[i]
		if srcDim == tgtDim {
			strides[i] = stride
			continue
		}
		if srcDim != 1 {
			return nil, errors.New("incompatible broadcast dimensions")
		}
		strides[i] = 0
	}
	out := &Tensor{
		data:         t.data,
		shape:        newShape,
		strides:      strides,
		requiresGrad: t.requiresGrad,
	}
	if t.requiresGrad {
		out.parents = []*Tensor{t}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				reduced, err := ReduceToShape(grad, srcShape)
				if err != nil {
					panic(err)
				}
				accumulate(grads, t, reduced)
			},
		}
	}
	return out, nil
}

func ReduceToShape(grad *Tensor, targetShape []int) (*Tensor, error) {
	tgt := append([]int(nil), targetShape...)
	if len(tgt) == 0 {
		tgt = []int{1}
	}
	if len(tgt) > len(grad.shape) {
		return nil, errors.New("target rank greater than grad rank")
	}
	out := grad
	diff := len(out.shape) - len(tgt)
	for axis := 0; axis < len(out.shape); axis++ {
		var tgtDim int
		if axis < diff {
			tgtDim = 1
		} else {
			tgtDim = tgt[axis-diff]
		}
		if out.shape[axis] == tgtDim {
			continue
		}
		if tgtDim != 1 {
			return nil, errors.New("cannot reduce to target shape")
		}
		out = reduceAxis(out, axis)
	}
	if len(out.shape) != len(tgt) {
		return reshapeKeep(out, tgt), nil
	}
	for i, dim := range tgt {
		if out.shape[i] != dim {
			return reshapeKeep(out, tgt), nil
		}
	}
	return out, nil
}

func reduceAxis(t *Tensor, axis int) *Tensor {
	if axis < 0 || axis >= len(t.shape) {
		panic("axis out of range")
	}
	shape := append([]int(nil), t.shape...)
	axisSize := shape[axis]
	shape[axis] = 1
	out := Zeros(shape...)
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= t.shape[i]
	}
	inner := 1
	for i := axis + 1; i < len(t.shape); i++ {
		inner *= t.shape[i]
	}
	parallel.For(outer, func(start, end int) {
		for o := start; o < end; o++ {
			dstBase := o * inner
			srcBase := o * axisSize * inner
			for k := 0; k < axisSize; k++ {
				srcOffset := srcBase + k*inner
				for j := 0; j < inner; j++ {
					out.data[dstBase+j] += t.data[srcOffset+j]
				}
			}
		}
	})
	return out
}

func reshapeKeep(t *Tensor, shape []int) *Tensor {
	tgt := append([]int(nil), shape...)
	if len(tgt) == 0 {
		tgt = []int{1}
	}
	total := 1
	for _, dim := range tgt {
		if dim <= 0 {
			panic("invalid reshape target")
		}
		total *= dim
	}
	if total != len(t.data) {
		panic("reshapeKeep size mismatch")
	}
	return &Tensor{
		data:    t.data,
		shape:   tgt,
		strides: makeStrides(tgt),
	}
}
