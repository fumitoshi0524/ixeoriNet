package optim

import (
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Constraint interface {
	Apply(param *tensor.Tensor) error
}

type MaxNormConstraint struct {
	maxNorm float64
	norm    float64
}

func NewMaxNormConstraint(maxNorm, norm float64) *MaxNormConstraint {
	if norm <= 0 {
		norm = 2
	}
	return &MaxNormConstraint{maxNorm: maxNorm, norm: norm}
}

func (c *MaxNormConstraint) Apply(param *tensor.Tensor) error {
	if param == nil || c.maxNorm <= 0 {
		return nil
	}
	sum := 0.0
	for _, v := range param.Data() {
		abs := math.Abs(v)
		sum += math.Pow(abs, c.norm)
	}
	norm := math.Pow(sum, 1.0/c.norm)
	if norm <= c.maxNorm {
		return nil
	}
	scale := c.maxNorm / (norm + 1e-12)
	param.Scale(scale)
	return nil
}
