package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func AddBias2D(a, bias *Tensor) (*Tensor, error) {
	if len(a.shape) != 2 {
		return nil, errors.New("AddBias2D expects rank 2 tensor input")
	}
	if len(bias.shape) != 1 {
		return nil, errors.New("AddBias2D expects rank 1 bias")
	}
	if a.shape[1] != bias.shape[0] {
		return nil, errors.New("AddBias2D dimension mismatch")
	}
	out := Zeros(a.shape...)
	copy(out.data, a.data)
	cols := a.shape[1]
	rows := a.shape[0]
	parallel.For(rows, func(start, end int) {
		for i := start; i < end; i++ {
			offset := i * cols
			for j := 0; j < cols; j++ {
				out.data[offset+j] += bias.data[j]
			}
		}
	})
	if a.requiresGrad || bias.requiresGrad {
		out.requiresGrad = true
		parents := make([]*Tensor, 0, 2)
		if a.requiresGrad {
			parents = append(parents, a)
		}
		if bias.requiresGrad {
			parents = append(parents, bias)
		}
		out.parents = parents
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				if a.requiresGrad {
					accumulate(grads, a, grad)
				}
				if bias.requiresGrad {
					agg := Zeros(bias.shape...)
					for i := 0; i < rows; i++ {
						offset := i * cols
						for j := 0; j < cols; j++ {
							agg.data[j] += grad.data[offset+j]
						}
					}
					accumulate(grads, bias, agg)
				}
			},
		}
	}
	return out, nil
}
