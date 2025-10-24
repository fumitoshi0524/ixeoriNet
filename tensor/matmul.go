package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func MatMul(a, b *Tensor) (*Tensor, error) {
	if len(a.shape) != 2 || len(b.shape) != 2 {
		return nil, errors.New("matmul expects rank 2 tensors")
	}
	aRows, aCols := a.shape[0], a.shape[1]
	bRows, bCols := b.shape[0], b.shape[1]
	if aCols != bRows {
		return nil, errors.New("incompatible shapes for matmul")
	}
	out := Zeros(aRows, bCols)
	parallel.For(aRows, func(start, end int) {
		for i := start; i < end; i++ {
			offsetOut := i * bCols
			offsetA := i * aCols
			for k := 0; k < aCols; k++ {
				aik := a.data[offsetA+k]
				offsetB := k * bCols
				for j := 0; j < bCols; j++ {
					out.data[offsetOut+j] += aik * b.data[offsetB+j]
				}
			}
		}
	})
	if a.requiresGrad || b.requiresGrad {
		out.requiresGrad = true
		parents := make([]*Tensor, 0, 2)
		if a.requiresGrad {
			parents = append(parents, a)
		}
		if b.requiresGrad {
			parents = append(parents, b)
		}
		out.parents = parents
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				if a.requiresGrad {
					ga := matmulRaw(grad, b, false, true)
					accumulate(grads, a, ga)
				}
				if b.requiresGrad {
					gb := matmulRaw(a, grad, true, false)
					accumulate(grads, b, gb)
				}
			},
		}
	}
	return out, nil
}

func matmulRaw(a, b *Tensor, transA, transB bool) *Tensor {
	aRows, aCols := shape2D(a, transA)
	bRows, bCols := shape2D(b, transB)
	if aCols != bRows {
		panic("matmulRaw shape mismatch")
	}
	out := Zeros(aRows, bCols)
	parallel.For(aRows, func(start, end int) {
		for i := start; i < end; i++ {
			for k := 0; k < aCols; k++ {
				aik := index2D(a, i, k, transA)
				for j := 0; j < bCols; j++ {
					out.data[i*bCols+j] += aik * index2D(b, k, j, transB)
				}
			}
		}
	})
	return out
}

func shape2D(t *Tensor, trans bool) (int, int) {
	if len(t.shape) != 2 {
		panic("shape2D expects rank 2 tensor")
	}
	if trans {
		return t.shape[1], t.shape[0]
	}
	return t.shape[0], t.shape[1]
}

func index2D(t *Tensor, row, col int, trans bool) float64 {
	if !trans {
		return t.data[row*t.shape[1]+col]
	}
	return t.data[col*t.shape[1]+row]
}
