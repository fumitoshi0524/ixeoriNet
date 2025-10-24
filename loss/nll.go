package loss

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

// NLLLoss computes the negative log likelihood loss given log-probabilities and target indices.
func NLLLoss(logProb *tensor.Tensor, target []int) (*tensor.Tensor, error) {
	shape := logProb.Shape()
	if len(shape) != 2 {
		return nil, errors.New("NLLLoss expects input shape [batch, classes]")
	}
	batch := shape[0]
	classes := shape[1]
	if len(target) != batch {
		return nil, errors.New("target length must equal batch size")
	}
	sum := 0.0
	for i := 0; i < batch; i++ {
		label := target[i]
		if label < 0 || label >= classes {
			return nil, errors.New("target index out of range")
		}
		idx := i*classes + label
		sum -= logProb.Data()[idx]
	}
	loss := tensor.Full(sum/float64(batch), 1)
	return loss, nil
}
