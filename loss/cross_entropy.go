package loss

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func CrossEntropy(logits *tensor.Tensor, targets []int) (*tensor.Tensor, error) {
	shape := logits.Shape()
	if len(shape) != 2 {
		return nil, errors.New("CrossEntropy expects rank 2 logits")
	}
	batch, classes := shape[0], shape[1]
	if len(targets) != batch {
		return nil, errors.New("target length mismatch")
	}
	data := make([]float64, batch*classes)
	for i, idx := range targets {
		if idx < 0 || idx >= classes {
			return nil, errors.New("target index out of range")
		}
		data[i*classes+idx] = 1
	}
	targetTensor := tensor.MustNew(data, batch, classes)
	logProb, err := tensor.LogSoftmax(logits, 1)
	if err != nil {
		return nil, err
	}
	masked, err := tensor.Mul(logProb, targetTensor)
	if err != nil {
		return nil, err
	}
	s := tensor.Sum(masked)
	sign := tensor.Full(-1, s.Shape()...)
	loss, err := tensor.Mul(s, sign)
	if err != nil {
		return nil, err
	}
	scale := tensor.Full(1.0/float64(batch), loss.Shape()...)
	loss, err = tensor.Mul(loss, scale)
	if err != nil {
		return nil, err
	}
	return loss, nil
}
