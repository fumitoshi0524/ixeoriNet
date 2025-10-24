package tensor

import (
	"errors"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func Add(a, b *Tensor) (*Tensor, error) {
	if err := ensureSameShape(a, b); err != nil {
		return nil, err
	}
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = a.data[i] + b.data[i]
		}
	})
	attachBinaryGrad(out, a, b, func(grad *Tensor, grads map[*Tensor]*Tensor, left, right *Tensor) {
		if left.requiresGrad {
			accumulate(grads, left, grad)
		}
		if right.requiresGrad {
			accumulate(grads, right, grad)
		}
	})
	return out, nil
}

func Sub(a, b *Tensor) (*Tensor, error) {
	if err := ensureSameShape(a, b); err != nil {
		return nil, err
	}
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = a.data[i] - b.data[i]
		}
	})
	attachBinaryGrad(out, a, b, func(grad *Tensor, grads map[*Tensor]*Tensor, left, right *Tensor) {
		if left.requiresGrad {
			accumulate(grads, left, grad)
		}
		if right.requiresGrad {
			tmp := grad.Clone()
			parallel.For(len(tmp.data), func(start, end int) {
				for i := start; i < end; i++ {
					tmp.data[i] = -tmp.data[i]
				}
			})
			accumulate(grads, right, tmp)
		}
	})
	return out, nil
}

func Mul(a, b *Tensor) (*Tensor, error) {
	if err := ensureSameShape(a, b); err != nil {
		return nil, err
	}
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = a.data[i] * b.data[i]
		}
	})
	attachBinaryGrad(out, a, b, func(grad *Tensor, grads map[*Tensor]*Tensor, left, right *Tensor) {
		if left.requiresGrad {
			accumulate(grads, left, hadamard(grad, right.Detach()))
		}
		if right.requiresGrad {
			accumulate(grads, right, hadamard(grad, left.Detach()))
		}
	})
	return out, nil
}

func Div(a, b *Tensor) (*Tensor, error) {
	if err := ensureSameShape(a, b); err != nil {
		return nil, err
	}
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = a.data[i] / b.data[i]
		}
	})
	attachBinaryGrad(out, a, b, func(grad *Tensor, grads map[*Tensor]*Tensor, left, right *Tensor) {
		if left.requiresGrad {
			accumulate(grads, left, hadamard(grad, reciprocal(right.Detach())))
		}
		if right.requiresGrad {
			numerator := hadamard(grad, left.Detach())
			parallel.For(len(numerator.data), func(start, end int) {
				for i := start; i < end; i++ {
					numerator.data[i] = -numerator.data[i] / (right.data[i] * right.data[i])
				}
			})
			accumulate(grads, right, numerator)
		}
	})
	return out, nil
}

func Pow(a *Tensor, value float64) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = math.Pow(a.data[i], value)
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				coef := value
				base := a.Detach()
				parallel.For(len(base.data), func(start, end int) {
					for i := start; i < end; i++ {
						base.data[i] = coef * math.Pow(base.data[i], value-1)
					}
				})
				accumulate(grads, a, hadamard(grad, base))
			},
		}
	}
	return out
}

func Exp(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = math.Exp(a.data[i])
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				accumulate(grads, a, hadamard(grad, out.Detach()))
			},
		}
	}
	return out
}

func Log(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = math.Log(a.data[i])
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				accumulate(grads, a, hadamard(grad, reciprocal(a.Detach())))
			},
		}
	}
	return out
}

func Sum(a *Tensor) *Tensor {
	val := 0.0
	for _, v := range a.data {
		val += v
	}
	out := MustNew([]float64{val}, 1)
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				expanded := Full(grad.data[0], a.shape...)
				accumulate(grads, a, expanded)
			},
		}
	}
	return out
}

func Mean(a *Tensor) *Tensor {
	s := Sum(a)
	scale := 1.0 / float64(a.Numel())
	parallel.For(len(s.data), func(start, end int) {
		for i := start; i < end; i++ {
			s.data[i] *= scale
		}
	})
	if a.requiresGrad {
		s.requiresGrad = true
		s.parents = []*Tensor{a}
		s.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				expanded := Full(grad.data[0]*scale, a.shape...)
				accumulate(grads, a, expanded)
			},
		}
	}
	return s
}

func hadamard(a, b *Tensor) *Tensor {
	if err := ensureSameShape(a, b); err != nil {
		panic(err)
	}
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = a.data[i] * b.data[i]
		}
	})
	return out
}

func reciprocal(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = 1.0 / a.data[i]
		}
	})
	return out
}

func attachBinaryGrad(out, a, b *Tensor, backward func(grad *Tensor, grads map[*Tensor]*Tensor, left, right *Tensor)) {
	if !(a.requiresGrad || b.requiresGrad) {
		return
	}
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
			backward(grad, grads, a, b)
		},
	}
}

func ensureSameShape(a, b *Tensor) error {
	if len(a.shape) != len(b.shape) {
		return errors.New("shape mismatch")
	}
	for i, dim := range a.shape {
		if dim != b.shape[i] {
			return errors.New("shape mismatch")
		}
	}
	return nil
}
