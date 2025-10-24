package nn

import (
	"fmt"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type BatchNorm struct {
	numFeatures int
	momentum    float64
	eps         float64
	affine      bool
	training    bool
	weight      *tensor.Tensor
	bias        *tensor.Tensor
	runningMean *tensor.Tensor
	runningVar  *tensor.Tensor
}

func NewBatchNorm(numFeatures int, momentum, eps float64, affine bool) *BatchNorm {
	if momentum <= 0 || momentum >= 1 {
		momentum = 0.1
	}
	if eps <= 0 {
		eps = 1e-5
	}
	weight := tensor.Ones(numFeatures)
	bias := tensor.Zeros(numFeatures)
	if affine {
		weight.SetRequiresGrad(true)
		bias.SetRequiresGrad(true)
	} else {
		weight = nil
		bias = nil
	}
	runningMean := tensor.Zeros(numFeatures)
	runningVar := tensor.Ones(numFeatures)
	return &BatchNorm{
		numFeatures: numFeatures,
		momentum:    momentum,
		eps:         eps,
		affine:      affine,
		training:    true,
		weight:      weight,
		bias:        bias,
		runningMean: runningMean,
		runningVar:  runningVar,
	}
}

func (bn *BatchNorm) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.BatchNorm(input, bn.runningMean, bn.runningVar, bn.weight, bn.bias, bn.momentum, bn.eps, bn.training)
}

func (bn *BatchNorm) Parameters() []*tensor.Tensor {
	if !bn.affine {
		return nil
	}
	return []*tensor.Tensor{bn.weight, bn.bias}
}

func (bn *BatchNorm) ZeroGrad() {
	if !bn.affine {
		return
	}
	bn.weight.ZeroGrad()
	bn.bias.ZeroGrad()
}

func (bn *BatchNorm) Train() {
	bn.training = true
}

func (bn *BatchNorm) Eval() {
	bn.training = false
}

func (bn *BatchNorm) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	if bn.affine {
		state[joinPrefix(prefix, "weight")] = bn.weight.Clone()
		state[joinPrefix(prefix, "bias")] = bn.bias.Clone()
	}
	state[joinPrefix(prefix, "running_mean")] = bn.runningMean.Clone()
	state[joinPrefix(prefix, "running_var")] = bn.runningVar.Clone()
}

func (bn *BatchNorm) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	if bn.affine {
		wKey := joinPrefix(prefix, "weight")
		w, ok := state[wKey]
		if !ok {
			return fmt.Errorf("BatchNorm missing %s", wKey)
		}
		if err := tensor.CopyInto(bn.weight, w); err != nil {
			return fmt.Errorf("load %s: %w", wKey, err)
		}
		bKey := joinPrefix(prefix, "bias")
		b, ok := state[bKey]
		if !ok {
			return fmt.Errorf("BatchNorm missing %s", bKey)
		}
		if err := tensor.CopyInto(bn.bias, b); err != nil {
			return fmt.Errorf("load %s: %w", bKey, err)
		}
	}
	meanKey := joinPrefix(prefix, "running_mean")
	mean, ok := state[meanKey]
	if !ok {
		return fmt.Errorf("BatchNorm missing %s", meanKey)
	}
	if err := tensor.CopyInto(bn.runningMean, mean); err != nil {
		return fmt.Errorf("load %s: %w", meanKey, err)
	}
	varKey := joinPrefix(prefix, "running_var")
	variance, ok := state[varKey]
	if !ok {
		return fmt.Errorf("BatchNorm missing %s", varKey)
	}
	if err := tensor.CopyInto(bn.runningVar, variance); err != nil {
		return fmt.Errorf("load %s: %w", varKey, err)
	}
	return nil
}

func (bn *BatchNorm) RunningMean() *tensor.Tensor {
	return bn.runningMean
}

func (bn *BatchNorm) RunningVar() *tensor.Tensor {
	return bn.runningVar
}
