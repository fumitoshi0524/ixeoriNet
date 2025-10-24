package nn

import (
	"fmt"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type LayerNorm struct {
	normalizedShape []int
	eps             float64
	affine          bool
	weight          *tensor.Tensor
	bias            *tensor.Tensor
}

func NewLayerNorm(normalizedShape []int, eps float64, affine bool) *LayerNorm {
	shapeCopy := append([]int(nil), normalizedShape...)
	if eps <= 0 {
		eps = 1e-5
	}
	var weight *tensor.Tensor
	var bias *tensor.Tensor
	if affine {
		weight = tensor.Ones(shapeCopy...)
		bias = tensor.Zeros(shapeCopy...)
		weight.SetRequiresGrad(true)
		bias.SetRequiresGrad(true)
	}
	return &LayerNorm{
		normalizedShape: shapeCopy,
		eps:             eps,
		affine:          affine,
		weight:          weight,
		bias:            bias,
	}
}

func (ln *LayerNorm) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.LayerNorm(input, ln.normalizedShape, ln.weight, ln.bias, ln.eps)
}

func (ln *LayerNorm) Parameters() []*tensor.Tensor {
	if !ln.affine {
		return nil
	}
	return []*tensor.Tensor{ln.weight, ln.bias}
}

func (ln *LayerNorm) ZeroGrad() {
	if !ln.affine {
		return
	}
	ln.weight.ZeroGrad()
	ln.bias.ZeroGrad()
}

func (ln *LayerNorm) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	if ln.affine {
		state[joinPrefix(prefix, "weight")] = ln.weight.Clone()
		state[joinPrefix(prefix, "bias")] = ln.bias.Clone()
	}
}

func (ln *LayerNorm) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	if ln.affine {
		wKey := joinPrefix(prefix, "weight")
		w, ok := state[wKey]
		if !ok {
			return fmt.Errorf("LayerNorm missing %s", wKey)
		}
		if err := tensor.CopyInto(ln.weight, w); err != nil {
			return fmt.Errorf("load %s: %w", wKey, err)
		}
		bKey := joinPrefix(prefix, "bias")
		b, ok := state[bKey]
		if !ok {
			return fmt.Errorf("LayerNorm missing %s", bKey)
		}
		if err := tensor.CopyInto(ln.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", bKey, err)
		}
	}
	return nil
}
