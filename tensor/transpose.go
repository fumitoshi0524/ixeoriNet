package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func Transpose(a *Tensor) (*Tensor, error) {
	if len(a.shape) != 2 {
		return nil, errors.New("transpose expects rank 2 tensor")
	}
	rows, cols := a.shape[0], a.shape[1]
	out := Zeros(cols, rows)
	parallel.For(rows, func(start, end int) {
		for i := start; i < end; i++ {
			offset := i * cols
			for j := 0; j < cols; j++ {
				out.data[j*rows+i] = a.data[offset+j]
			}
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				accumulate(grads, a, grad.MustTranspose())
			},
		}
	}
	return out, nil
}

func (t *Tensor) MustTranspose() *Tensor {
	tr, err := Transpose(t)
	if err != nil {
		panic(err)
	}
	return tr
}
