package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func Max(a *Tensor, axis int) (*Tensor, error) {
	return reduceMaxMin(a, axis, true)
}

func Min(a *Tensor, axis int) (*Tensor, error) {
	return reduceMaxMin(a, axis, false)
}

func reduceMaxMin(a *Tensor, axis int, isMax bool) (*Tensor, error) {
	if len(a.shape) == 0 {
		return nil, errors.New("reduction requires rank >= 1 tensor")
	}
	rank := len(a.shape)
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, errors.New("axis out of range")
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= a.shape[i]
	}
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= a.shape[i]
	}
	axisSize := a.shape[axis]
	if axisSize == 0 {
		return nil, errors.New("cannot reduce over zero-sized axis")
	}
	outShape := make([]int, 0, rank-1)
	for i, dim := range a.shape {
		if i == axis {
			continue
		}
		outShape = append(outShape, dim)
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	out := Zeros(outShape...)
	indices := make([]int, len(out.data))
	parallel.For(outer, func(start, end int) {
		for o := start; o < end; o++ {
			dstBase := o * inner
			srcBase := o * axisSize * inner
			for in := 0; in < inner; in++ {
				bestIdx := 0
				bestVal := a.data[srcBase+in]
				for k := 1; k < axisSize; k++ {
					candidate := a.data[srcBase+k*inner+in]
					if isMax {
						if candidate > bestVal {
							bestVal = candidate
							bestIdx = k
						}
					} else {
						if candidate < bestVal {
							bestVal = candidate
							bestIdx = k
						}
					}
				}
				outIndex := dstBase + in
				out.data[outIndex] = bestVal
				indices[outIndex] = bestIdx
			}
		}
	})
	if !a.requiresGrad {
		return out, nil
	}
	out.requiresGrad = true
	out.parents = []*Tensor{a}
	axisSizeCopy := axisSize
	innerCopy := inner
	indicesCopy := append([]int(nil), indices...)
	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			g := Zeros(a.shape...)
			parallel.For(len(indicesCopy), func(start, end int) {
				for idx := start; idx < end; idx++ {
					outerIdx := 0
					if innerCopy > 0 {
						outerIdx = idx / innerCopy
					}
					innerIdx := 0
					if innerCopy > 0 {
						innerIdx = idx % innerCopy
					}
					dst := outerIdx*axisSizeCopy*innerCopy + indicesCopy[idx]*innerCopy + innerIdx
					g.data[dst] += grad.data[idx]
				}
			})
			accumulate(grads, a, g)
		},
	}
	return out, nil
}

// SumAxis sums elements along the given axis and returns a tensor with that
// axis removed. Behaves similarly to Max/Min and preserves autograd when
// the input requires gradients.
func SumAxis(a *Tensor, axis int) (*Tensor, error) {
	if len(a.shape) == 0 {
		return nil, errors.New("reduction requires rank >= 1 tensor")
	}
	rank := len(a.shape)
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, errors.New("axis out of range")
	}
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= a.shape[i]
	}
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= a.shape[i]
	}
	axisSize := a.shape[axis]
	if axisSize == 0 {
		return nil, errors.New("cannot reduce over zero-sized axis")
	}
	outShape := make([]int, 0, rank-1)
	for i, dim := range a.shape {
		if i == axis {
			continue
		}
		outShape = append(outShape, dim)
	}
	if len(outShape) == 0 {
		outShape = []int{1}
	}
	out := Zeros(outShape...)
	parallel.For(outer, func(start, end int) {
		for o := start; o < end; o++ {
			dstBase := o * inner
			srcBase := o * axisSize * inner
			for in := 0; in < inner; in++ {
				s := 0.0
				for k := 0; k < axisSize; k++ {
					s += a.data[srcBase+k*inner+in]
				}
				outIndex := dstBase + in
				out.data[outIndex] = s
			}
		}
	})
	if !a.requiresGrad {
		return out, nil
	}
	out.requiresGrad = true
	out.parents = []*Tensor{a}
	axisSizeCopy := axisSize
	innerCopy := inner
	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			g := Zeros(a.shape...)
			parallel.For(outer*innerCopy, func(start, end int) {
				for idx := start; idx < end; idx++ {
					outerIdx := 0
					if innerCopy > 0 {
						outerIdx = idx / innerCopy
					}
					innerIdx := 0
					if innerCopy > 0 {
						innerIdx = idx % innerCopy
					}
					base := outerIdx*axisSizeCopy*innerCopy + innerIdx
					for k := 0; k < axisSizeCopy; k++ {
						g.data[base+k*innerCopy] += grad.data[idx]
					}
				}
			})
			accumulate(grads, a, g)
		},
	}
	return out, nil
}

// MeanAxis computes the mean along the given axis and returns a tensor with
// that axis removed. It reuses SumAxis and scales the result appropriately.
func MeanAxis(a *Tensor, axis int) (*Tensor, error) {
	s, err := SumAxis(a, axis)
	if err != nil {
		return nil, err
	}
	axisSize := a.shape[axis]
	scale := 1.0 / float64(axisSize)
	scaleT := Full(scale, s.Shape()...)
	scaled, err := Mul(s, scaleT)
	if err != nil {
		return nil, err
	}
	return scaled, nil
}
