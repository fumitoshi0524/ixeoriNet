package tensor

import "errors"

// Conv1D performs 1D convolution over input [batch, channels, width].
func Conv1D(input, weight, bias *Tensor, stride, pad int) (*Tensor, error) {
	if len(input.shape) != 3 {
		return nil, errors.New("Conv1D expects input shape [batch, channels, width]")
	}
	if len(weight.shape) != 3 {
		return nil, errors.New("Conv1D expects weight shape [out_channels, in_channels, kernel_w]")
	}
	if bias != nil && len(bias.shape) != 1 {
		return nil, errors.New("bias for Conv1D must be rank 1")
	}
	batch := input.shape[0]
	inChannels := input.shape[1]
	inW := input.shape[2]
	outChannels := weight.shape[0]
	kernelChannels := weight.shape[1]
	kernelW := weight.shape[2]
	if kernelChannels != inChannels {
		return nil, errors.New("kernel in_channels mismatch")
	}
	if stride <= 0 {
		return nil, errors.New("stride must be positive")
	}
	outW := (inW+2*pad-kernelW)/stride + 1
	if outW <= 0 {
		return nil, errors.New("invalid output size")
	}
	out := Zeros(batch, outChannels, outW)
	for n := 0; n < batch; n++ {
		for oc := 0; oc < outChannels; oc++ {
			for ow := 0; ow < outW; ow++ {
				acc := 0.0
				for ic := 0; ic < inChannels; ic++ {
					for kw := 0; kw < kernelW; kw++ {
						iw := ow*stride - pad + kw
						if iw < 0 || iw >= inW {
							continue
						}
						inputIdx := ((n*inChannels+ic)*inW + iw)
						weightIdx := ((oc*inChannels+ic)*kernelW + kw)
						acc += input.data[inputIdx] * weight.data[weightIdx]
					}
				}
				if bias != nil {
					acc += bias.data[oc]
				}
				out.data[(n*outChannels+oc)*outW+ow] = acc
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
					for oc := 0; oc < outChannels; oc++ {
						for ow := 0; ow < outW; ow++ {
							gVal := grad.data[(n*outChannels+oc)*outW+ow]
							for ic := 0; ic < inChannels; ic++ {
								for kw := 0; kw < kernelW; kw++ {
									iw := ow*stride - pad + kw
									if iw < 0 || iw >= inW {
										continue
									}
									inputIdx := ((n*inChannels+ic)*inW + iw)
									weightIdx := ((oc*inChannels+ic)*kernelW + kw)
									gInput.data[inputIdx] += weight.data[weightIdx] * gVal
								}
							}
						}
					}
				}
				accumulate(grads, input, gInput)
			}
			if weight.requiresGrad {
				gWeight := Zeros(weight.shape...)
				for n := 0; n < batch; n++ {
					for oc := 0; oc < outChannels; oc++ {
						for ow := 0; ow < outW; ow++ {
							gVal := grad.data[(n*outChannels+oc)*outW+ow]
							for ic := 0; ic < inChannels; ic++ {
								for kw := 0; kw < kernelW; kw++ {
									iw := ow*stride - pad + kw
									if iw < 0 || iw >= inW {
										continue
									}
									inputIdx := ((n*inChannels+ic)*inW + iw)
									weightIdx := ((oc*inChannels+ic)*kernelW + kw)
									gWeight.data[weightIdx] += input.data[inputIdx] * gVal
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
							gBias.data[oc] += grad.data[(n*outChannels+oc)*outW+ow]
						}
					}
				}
				accumulate(grads, bias, gBias)
			}
		},
	}
	return out, nil
}
