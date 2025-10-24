package loss

import "github.com/fumitoshi0524/ixeoriNet/tensor"

func MSE(pred, target *tensor.Tensor) (*tensor.Tensor, error) {
	diff, err := tensor.Sub(pred, target)
	if err != nil {
		return nil, err
	}
	sq := tensor.Pow(diff, 2)
	return tensor.Mean(sq), nil
}
