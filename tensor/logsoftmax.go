package tensor

import (
	"errors"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func LogSoftmax(a *Tensor, axis int) (*Tensor, error) {
	if len(a.shape) != 2 {
		return nil, errors.New("LogSoftmax expects rank 2 tensor")
	}
	if axis < 0 {
		axis += len(a.shape)
	}
	if axis != 1 {
		return nil, errors.New("LogSoftmax currently supports axis 1 only")
	}
	rows, cols := a.shape[0], a.shape[1]
	out := Zeros(rows, cols)
	parallel.For(rows, func(start, end int) {
		for i := start; i < end; i++ {
			offset := i * cols
			maxVal := a.data[offset]
			for j := 1; j < cols; j++ {
				v := a.data[offset+j]
				if v > maxVal {
					maxVal = v
				}
			}
			sum := 0.0
			for j := 0; j < cols; j++ {
				sum += math.Exp(a.data[offset+j] - maxVal)
			}
			logSum := maxVal + math.Log(sum)
			for j := 0; j < cols; j++ {
				out.data[offset+j] = a.data[offset+j] - logSum
			}
		}
	})
	if a.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{a}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				gx := Zeros(a.shape...)
				parallel.For(rows, func(start, end int) {
					for i := start; i < end; i++ {
						offset := i * cols
						sumGrad := 0.0
						for j := 0; j < cols; j++ {
							sumGrad += grad.data[offset+j]
						}
						for j := 0; j < cols; j++ {
							soft := math.Exp(out.data[offset+j])
							gx.data[offset+j] = grad.data[offset+j] - soft*sumGrad
						}
					}
				})
				accumulate(grads, a, gx)
			},
		}
	}
	return out, nil
}

func Softmax(a *Tensor, axis int) (*Tensor, error) {
	logsm, err := LogSoftmax(a, axis)
	if err != nil {
		return nil, err
	}
	return Exp(logsm), nil
}
