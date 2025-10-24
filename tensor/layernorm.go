package tensor

import (
	"errors"
	"math"
)

// LayerNorm normalizes the last len(normalizedShape) dimensions of the input.
func LayerNorm(input *Tensor, normalizedShape []int, weight, bias *Tensor, eps float64) (*Tensor, error) {
	if len(normalizedShape) == 0 {
		return nil, errors.New("normalized shape required")
	}
	rank := len(input.shape)
	normRank := len(normalizedShape)
	if normRank > rank {
		return nil, errors.New("normalized shape rank exceeds input rank")
	}
	for i := 0; i < normRank; i++ {
		if input.shape[rank-normRank+i] != normalizedShape[i] {
			return nil, errors.New("normalized shape mismatch")
		}
	}
	normSize := 1
	for _, dim := range normalizedShape {
		if dim <= 0 {
			return nil, errors.New("invalid normalized dimension")
		}
		normSize *= dim
	}
	if weight != nil && weight.Numel() != normSize {
		return nil, errors.New("weight size mismatch")
	}
	if bias != nil && bias.Numel() != normSize {
		return nil, errors.New("bias size mismatch")
	}
	if eps <= 0 {
		eps = 1e-5
	}

	out := Zeros(input.shape...)
	outer := input.Numel() / normSize
	savedMean := make([]float64, outer)
	savedInvStd := make([]float64, outer)
	xhat := make([]float64, input.Numel())

	for o := 0; o < outer; o++ {
		offset := o * normSize
		sum := 0.0
		for j := 0; j < normSize; j++ {
			sum += input.data[offset+j]
		}
		mean := sum / float64(normSize)
		savedMean[o] = mean
		varSum := 0.0
		for j := 0; j < normSize; j++ {
			diff := input.data[offset+j] - mean
			varSum += diff * diff
		}
		invStd := 1.0 / math.Sqrt(varSum/float64(normSize)+eps)
		savedInvStd[o] = invStd
		for j := 0; j < normSize; j++ {
			idx := offset + j
			xh := (input.data[idx] - mean) * invStd
			xhat[idx] = xh
			val := xh
			if weight != nil {
				val *= weight.data[j]
			}
			if bias != nil {
				val += bias.data[j]
			}
			out.data[idx] = val
		}
	}

	requireGrad := input.requiresGrad || (weight != nil && weight.requiresGrad) || (bias != nil && bias.requiresGrad)
	if !requireGrad {
		return out, nil
	}

	parents := make([]*Tensor, 0, 3)
	if input.requiresGrad {
		parents = append(parents, input)
	}
	if weight != nil && weight.requiresGrad {
		parents = append(parents, weight)
	}
	if bias != nil && bias.requiresGrad {
		parents = append(parents, bias)
	}

	out.requiresGrad = true
	out.parents = parents
	normSizeF := float64(normSize)
	weightData := []float64(nil)
	if weight != nil {
		weightData = append([]float64(nil), weight.data...)
	}

	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			var gInput *Tensor
			if input.requiresGrad {
				gInput = Zeros(input.shape...)
			}
			var gWeight *Tensor
			var gBias *Tensor
			if weight != nil && weight.requiresGrad {
				gWeight = Zeros(weight.shape...)
			}
			if bias != nil && bias.requiresGrad {
				gBias = Zeros(bias.shape...)
			}

			for o := 0; o < outer; o++ {
				offset := o * normSize
				sumGrad := 0.0
				sumGradXhat := 0.0
				for j := 0; j < normSize; j++ {
					idx := offset + j
					gVal := grad.data[idx]
					scaled := gVal
					if weight != nil {
						scaled *= weightData[j]
					}
					sumGrad += scaled
					sumGradXhat += scaled * xhat[idx]
					if gWeight != nil {
						gWeight.data[j] += gVal * xhat[idx]
					}
					if gBias != nil {
						gBias.data[j] += gVal
					}
				}
				if gInput != nil {
					invStd := savedInvStd[o]
					for j := 0; j < normSize; j++ {
						idx := offset + j
						scaled := grad.data[idx]
						if weight != nil {
							scaled *= weightData[j]
						}
						xh := xhat[idx]
						term := scaled - sumGrad/normSizeF - xh*sumGradXhat/normSizeF
						gInput.data[idx] += term * invStd
					}
				}
			}

			if gInput != nil {
				accumulate(grads, input, gInput)
			}
			if gWeight != nil {
				accumulate(grads, weight, gWeight)
			}
			if gBias != nil {
				accumulate(grads, bias, gBias)
			}
		},
	}

	return out, nil
}
