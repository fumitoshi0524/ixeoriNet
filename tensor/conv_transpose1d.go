package tensor

import "errors"

// ConvTranspose1D performs a 1D transposed convolution (deconvolution).
// Input shape: [batch, in_channels, width]
// Weight shape: [in_channels, out_channels, kernel]
func ConvTranspose1D(input, weight, bias *Tensor, stride, padding int) (*Tensor, error) {
	if len(input.shape) != 3 {
		return nil, errors.New("ConvTranspose1D expects input shape [batch, channels, width]")
	}
	if len(weight.shape) != 3 {
		return nil, errors.New("ConvTranspose1D expects weight shape [in_channels, out_channels, kernel]")
	}
	if bias != nil && len(bias.shape) != 1 {
		return nil, errors.New("bias for ConvTranspose1D must be rank 1")
	}
	if stride <= 0 {
		return nil, errors.New("stride must be positive")
	}

	batch := input.shape[0]
	inChannels := input.shape[1]
	inW := input.shape[2]
	weightInChannels := weight.shape[0]
	outChannels := weight.shape[1]
	kernel := weight.shape[2]

	if weightInChannels != inChannels {
		return nil, errors.New("weight in_channels mismatch")
	}

	outW := (inW-1)*stride - 2*padding + kernel
	if outW <= 0 {
		return nil, errors.New("invalid output size")
	}

	out := Zeros(batch, outChannels, outW)
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for iw := 0; iw < inW; iw++ {
				inputVal := input.data[(n*inChannels+ic)*inW+iw]
				if inputVal == 0 {
					continue
				}
				base := iw*stride - padding
				for oc := 0; oc < outChannels; oc++ {
					for k := 0; k < kernel; k++ {
						ow := base + k
						if ow < 0 || ow >= outW {
							continue
						}
						weightIdx := (ic*outChannels+oc)*kernel + k
						outIdx := (n*outChannels+oc)*outW + ow
						out.data[outIdx] += inputVal * weight.data[weightIdx]
					}
				}
			}
		}
	}

	if bias != nil {
		for n := 0; n < batch; n++ {
			for oc := 0; oc < outChannels; oc++ {
				biasVal := bias.data[oc]
				if biasVal == 0 {
					continue
				}
				for ow := 0; ow < outW; ow++ {
					outIdx := (n*outChannels+oc)*outW + ow
					out.data[outIdx] += biasVal
				}
			}
		}
	}

	if !(input.requiresGrad || weight.requiresGrad || (bias != nil && bias.requiresGrad)) {
		return out, nil
	}

	parents := make([]*Tensor, 0, 3)
	if input.requiresGrad {
		parents = append(parents, input)
	}
	if weight.requiresGrad {
		parents = append(parents, weight)
	}
	if bias != nil && bias.requiresGrad {
		parents = append(parents, bias)
	}

	out.requiresGrad = true
	out.parents = parents
	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			if input.requiresGrad {
				gInput := Zeros(input.shape...)
				for n := 0; n < batch; n++ {
					for ic := 0; ic < inChannels; ic++ {
						for iw := 0; iw < inW; iw++ {
							sum := 0.0
							base := iw*stride - padding
							for oc := 0; oc < outChannels; oc++ {
								for k := 0; k < kernel; k++ {
									ow := base + k
									if ow < 0 || ow >= outW {
										continue
									}
									gradIdx := (n*outChannels+oc)*outW + ow
									weightIdx := (ic*outChannels+oc)*kernel + k
									sum += grad.data[gradIdx] * weight.data[weightIdx]
								}
							}
							gInput.data[(n*inChannels+ic)*inW+iw] = sum
						}
					}
				}
				accumulate(grads, input, gInput)
			}
			if weight.requiresGrad {
				gWeight := Zeros(weight.shape...)
				for n := 0; n < batch; n++ {
					for ic := 0; ic < inChannels; ic++ {
						for iw := 0; iw < inW; iw++ {
							inputVal := input.data[(n*inChannels+ic)*inW+iw]
							if inputVal == 0 {
								continue
							}
							base := iw*stride - padding
							for oc := 0; oc < outChannels; oc++ {
								for k := 0; k < kernel; k++ {
									ow := base + k
									if ow < 0 || ow >= outW {
										continue
									}
									gradIdx := (n*outChannels+oc)*outW + ow
									weightIdx := (ic*outChannels+oc)*kernel + k
									gWeight.data[weightIdx] += grad.data[gradIdx] * inputVal
								}
							}
						}
					}
				}
				accumulate(grads, weight, gWeight)
			}
			if bias != nil && bias.requiresGrad {
				gBias := Zeros(bias.shape...)
				for n := 0; n < batch; n++ {
					for oc := 0; oc < outChannels; oc++ {
						for ow := 0; ow < outW; ow++ {
							gradIdx := (n*outChannels+oc)*outW + ow
							gBias.data[oc] += grad.data[gradIdx]
						}
					}
				}
				accumulate(grads, bias, gBias)
			}
		},
	}

	return out, nil
}
