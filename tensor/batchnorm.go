package tensor

import (
	"errors"
	"math"
)

// BatchNorm applies batch normalization to inputs.
// Supports 2D ([batch, features]) and 4D ([batch, channels, H, W]) tensors.
func BatchNorm(input, runningMean, runningVar, weight, bias *Tensor, momentum, eps float64, training bool) (*Tensor, error) {
	if input == nil {
		return nil, errors.New("BatchNorm requires input tensor")
	}
	rank := len(input.shape)
	if rank != 2 && rank != 4 {
		return nil, errors.New("BatchNorm supports rank 2 or 4 tensors")
	}
	channels := input.shape[1]
	if runningMean != nil && len(runningMean.shape) != 1 {
		return nil, errors.New("running mean must be rank 1")
	}
	if runningVar != nil && len(runningVar.shape) != 1 {
		return nil, errors.New("running var must be rank 1")
	}
	if weight != nil && len(weight.shape) != 1 {
		return nil, errors.New("weight must be rank 1")
	}
	if bias != nil && len(bias.shape) != 1 {
		return nil, errors.New("bias must be rank 1")
	}

	o := Zeros(input.shape...)
	mean := make([]float64, channels)
	varVals := make([]float64, channels)
	invStd := make([]float64, channels)

	countPerChannel := input.shape[0]
	if rank == 4 {
		countPerChannel *= input.shape[2] * input.shape[3]
	}
	count := float64(countPerChannel)

	if training {
		// compute mean
		for n := 0; n < input.shape[0]; n++ {
			if rank == 2 {
				for c := 0; c < channels; c++ {
					idx := n*channels + c
					mean[c] += input.data[idx]
				}
				continue
			}
			for c := 0; c < channels; c++ {
				for h := 0; h < input.shape[2]; h++ {
					for w := 0; w < input.shape[3]; w++ {
						idx := ((n*channels+c)*input.shape[2]+h)*input.shape[3] + w
						mean[c] += input.data[idx]
					}
				}
			}
		}
		for c := 0; c < channels; c++ {
			mean[c] /= count
		}
		// compute variance
		for n := 0; n < input.shape[0]; n++ {
			if rank == 2 {
				for c := 0; c < channels; c++ {
					idx := n*channels + c
					diff := input.data[idx] - mean[c]
					varVals[c] += diff * diff
				}
				continue
			}
			for c := 0; c < channels; c++ {
				for h := 0; h < input.shape[2]; h++ {
					for w := 0; w < input.shape[3]; w++ {
						idx := ((n*channels+c)*input.shape[2]+h)*input.shape[3] + w
						diff := input.data[idx] - mean[c]
						varVals[c] += diff * diff
					}
				}
			}
		}
		for c := 0; c < channels; c++ {
			varVals[c] /= count
			invStd[c] = 1.0 / math.Sqrt(varVals[c]+eps)
			if runningMean != nil {
				runningMean.data[c] = (1-momentum)*runningMean.data[c] + momentum*mean[c]
			}
			if runningVar != nil {
				runningVar.data[c] = (1-momentum)*runningVar.data[c] + momentum*varVals[c]
			}
		}
	} else {
		if runningMean == nil || runningVar == nil {
			return nil, errors.New("BatchNorm eval requires running statistics")
		}
		copy(mean, runningMean.data)
		for c := 0; c < channels; c++ {
			invStd[c] = 1.0 / math.Sqrt(runningVar.data[c]+eps)
		}
	}

	apply := func(index int, channel int) {
		norm := (input.data[index] - mean[channel]) * invStd[channel]
		if weight != nil {
			norm *= weight.data[channel]
		}
		if bias != nil {
			norm += bias.data[channel]
		}
		o.data[index] = norm
	}

	if rank == 2 {
		for n := 0; n < input.shape[0]; n++ {
			for c := 0; c < channels; c++ {
				apply(n*channels+c, c)
			}
		}
	} else {
		for n := 0; n < input.shape[0]; n++ {
			for c := 0; c < channels; c++ {
				for h := 0; h < input.shape[2]; h++ {
					for w := 0; w < input.shape[3]; w++ {
						idx := ((n*channels+c)*input.shape[2]+h)*input.shape[3] + w
						apply(idx, c)
					}
				}
			}
		}
	}

	if !(input.requiresGrad || (weight != nil && weight.requiresGrad) || (bias != nil && bias.requiresGrad)) {
		return o, nil
	}

	savedMean := append([]float64(nil), mean...)
	savedInvStd := append([]float64(nil), invStd...)
	savedCount := count
	hasWeight := weight != nil
	hasBias := bias != nil
	var weightData []float64
	if weight != nil {
		weightData = append([]float64(nil), weight.data...)
	}

	o.requiresGrad = true
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
	o.parents = parents
	o.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			sumGrad := make([]float64, channels)
			sumGradXhat := make([]float64, channels)
			sumGradOrig := make([]float64, channels)
			sumGradOrigXhat := make([]float64, channels)

			evaluate := func(idx int, c int) (float64, float64, float64) {
				x := input.data[idx]
				goVal := grad.data[idx]
				scaled := goVal
				if hasWeight {
					scaled *= weightData[c]
				}
				xhat := (x - savedMean[c]) * savedInvStd[c]
				sumGrad[c] += scaled
				sumGradXhat[c] += scaled * xhat
				sumGradOrig[c] += goVal
				sumGradOrigXhat[c] += goVal * xhat
				return scaled, xhat, goVal
			}

			if rank == 2 {
				for n := 0; n < input.shape[0]; n++ {
					for c := 0; c < channels; c++ {
						idx := n*channels + c
						evaluate(idx, c)
					}
				}
			} else {
				for n := 0; n < input.shape[0]; n++ {
					for c := 0; c < channels; c++ {
						for h := 0; h < input.shape[2]; h++ {
							for w := 0; w < input.shape[3]; w++ {
								idx := ((n*channels+c)*input.shape[2]+h)*input.shape[3] + w
								evaluate(idx, c)
							}
						}
					}
				}
			}

			if input.requiresGrad {
				gInput := Zeros(input.shape...)
				if rank == 2 {
					for n := 0; n < input.shape[0]; n++ {
						for c := 0; c < channels; c++ {
							idx := n*channels + c
							goVal := grad.data[idx]
							scaled := goVal
							if hasWeight {
								scaled *= weightData[c]
							}
							xhat := (input.data[idx] - savedMean[c]) * savedInvStd[c]
							temp := scaled - sumGrad[c]/savedCount - xhat*sumGradXhat[c]/savedCount
							gInput.data[idx] = temp * savedInvStd[c]
						}
					}
				} else {
					for n := 0; n < input.shape[0]; n++ {
						for c := 0; c < channels; c++ {
							for h := 0; h < input.shape[2]; h++ {
								for w := 0; w < input.shape[3]; w++ {
									idx := ((n*channels+c)*input.shape[2]+h)*input.shape[3] + w
									goVal := grad.data[idx]
									scaled := goVal
									if hasWeight {
										scaled *= weightData[c]
									}
									xhat := (input.data[idx] - savedMean[c]) * savedInvStd[c]
									temp := scaled - sumGrad[c]/savedCount - xhat*sumGradXhat[c]/savedCount
									gInput.data[idx] = temp * savedInvStd[c]
								}
							}
						}
					}
				}
				accumulate(grads, input, gInput)
			}

			if hasWeight && weight.requiresGrad {
				gWeight := Zeros(weight.shape...)
				for c := 0; c < channels; c++ {
					gWeight.data[c] = sumGradOrigXhat[c]
				}
				accumulate(grads, weight, gWeight)
			}

			if hasBias && bias.requiresGrad {
				gBias := Zeros(bias.shape...)
				for c := 0; c < channels; c++ {
					gBias.data[c] = sumGradOrig[c]
				}
				accumulate(grads, bias, gBias)
			}
		},
	}

	return o, nil
}
