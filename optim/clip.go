package optim

import (
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func ClipGradNorm(params []*tensor.Tensor, maxNorm float64, normType float64) float64 {
	if maxNorm <= 0 {
		return 0
	}
	if normType <= 0 {
		normType = 2
	}
	total := 0.0
	for _, p := range params {
		if p == nil {
			continue
		}
		total += p.GradPowSum(normType)
	}
	norm := math.Pow(total, 1.0/normType)
	if norm > maxNorm && norm > 0 {
		scale := maxNorm / norm
		for _, p := range params {
			if p == nil {
				continue
			}
			p.ScaleGrad(scale)
		}
	}
	return norm
}

func ClipGradValue(params []*tensor.Tensor, clipValue float64) {
	if clipValue <= 0 {
		return
	}
	for _, p := range params {
		if p == nil {
			continue
		}
		p.ClipGradValue(clipValue)
	}
}
