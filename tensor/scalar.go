package tensor

import "github.com/fumitoshi0524/ixeoriNet/internal/parallel"

func AddScalar(a *Tensor, value float64) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = a.data[i] + value
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				accumulate(grads, a, grad)
			},
		}
	}
	return out
}

func MulScalar(a *Tensor, value float64) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = a.data[i] * value
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				scaled := grad.Clone()
				scaled.Scale(value)
				accumulate(grads, a, scaled)
			},
		}
	}
	return out
}
