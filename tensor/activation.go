package tensor

import (
	"math"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func Relu(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			v := a.data[i]
			if v > 0 {
				out.data[i] = v
			}
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		mask := out.Detach()
		parallel.For(len(mask.data), func(start, end int) {
			for i := start; i < end; i++ {
				if mask.data[i] > 0 {
					mask.data[i] = 1
				} else {
					mask.data[i] = 0
				}
			}
		})
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				accumulate(grads, a, hadamard(grad, mask))
			},
		}
	}
	return out
}

func Sigmoid(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = 1 / (1 + math.Exp(-a.data[i]))
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				oneMinus := Full(1, out.shape...)
				parallel.For(len(oneMinus.data), func(start, end int) {
					for i := start; i < end; i++ {
						oneMinus.data[i] -= out.data[i]
					}
				})
				accumulate(grads, a, hadamard(grad, hadamard(out.Detach(), oneMinus)))
			},
		}
	}
	return out
}

func Tanh(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			out.data[i] = math.Tanh(a.data[i])
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				one := Full(1, out.shape...)
				parallel.For(len(one.data), func(start, end int) {
					for i := start; i < end; i++ {
						t := out.data[i]
						one.data[i] -= t * t
					}
				})
				accumulate(grads, a, hadamard(grad, one))
			},
		}
	}
	return out
}

func LeakyRelu(a *Tensor, alpha float64) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			v := a.data[i]
			if v > 0 {
				out.data[i] = v
			} else {
				out.data[i] = alpha * v
			}
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		mask := Full(alpha, a.shape...)
		parallel.For(len(mask.data), func(start, end int) {
			for i := start; i < end; i++ {
				if a.data[i] > 0 {
					mask.data[i] = 1
				} else {
					mask.data[i] = alpha
				}
			}
		})
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				accumulate(grads, a, hadamard(grad, mask))
			},
		}
	}
	return out
}

func ELU(a *Tensor, alpha float64) *Tensor {
	out := Zeros(a.shape...)
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			v := a.data[i]
			if v > 0 {
				out.data[i] = v
			} else {
				out.data[i] = alpha * (math.Exp(v) - 1)
			}
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				factor := Zeros(a.shape...)
				parallel.For(len(factor.data), func(start, end int) {
					for i := start; i < end; i++ {
						if a.data[i] > 0 {
							factor.data[i] = 1
						} else {
							factor.data[i] = out.data[i] + alpha
						}
					}
				})
				accumulate(grads, a, hadamard(grad, factor))
			},
		}
	}
	return out
}

func Softplus(a *Tensor, beta float64) *Tensor {
	if beta <= 0 {
		beta = 1
	}
	out := Zeros(a.shape...)
	invBeta := 1 / beta
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			v := a.data[i] * beta
			out.data[i] = math.Log1p(math.Exp(v)) * invBeta
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				sigma := Zeros(a.shape...)
				parallel.For(len(sigma.data), func(start, end int) {
					for i := start; i < end; i++ {
						v := a.data[i] * beta
						sigma.data[i] = 1 / (1 + math.Exp(-v))
					}
				})
				accumulate(grads, a, hadamard(grad, sigma))
			},
		}
	}
	return out
}

func GELU(a *Tensor) *Tensor {
	out := Zeros(a.shape...)
	invSqrt2 := 1 / math.Sqrt2
	parallel.For(len(out.data), func(start, end int) {
		for i := start; i < end; i++ {
			v := a.data[i]
			out.data[i] = 0.5 * v * (1 + math.Erf(v*invSqrt2))
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		invSqrt2Pi := 1 / math.Sqrt(2*math.Pi)
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				factor := Zeros(a.shape...)
				parallel.For(len(factor.data), func(start, end int) {
					for i := start; i < end; i++ {
						v := a.data[i]
						erfTerm := math.Erf(v * invSqrt2)
						expTerm := math.Exp(-0.5 * v * v)
						factor.data[i] = 0.5*(1+erfTerm) + v*expTerm*invSqrt2Pi
					}
				})
				accumulate(grads, a, hadamard(grad, factor))
			},
		}
	}
	return out
}
