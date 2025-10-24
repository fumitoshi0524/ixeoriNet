package tensor

import (
	"math"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func (t *Tensor) GradPowSum(norm float64) float64 {
	if t == nil || t.grad == nil {
		return 0
	}
	sum := 0.0
	for _, v := range t.grad.data {
		abs := math.Abs(v)
		sum += math.Pow(abs, norm)
	}
	return sum
}

func (t *Tensor) ScaleGrad(factor float64) {
	if t == nil || t.grad == nil {
		return
	}
	t.grad.Scale(factor)
}

func (t *Tensor) ClipGradValue(limit float64) {
	if t == nil || t.grad == nil {
		return
	}
	if limit <= 0 {
		return
	}
	grad := t.grad
	parallel.For(len(grad.data), func(start, end int) {
		for i := start; i < end; i++ {
			v := grad.data[i]
			if v > limit {
				grad.data[i] = limit
			} else if v < -limit {
				grad.data[i] = -limit
			}
		}
	})
}
