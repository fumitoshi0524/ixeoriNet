package tensor

import "errors"

// Conv2D performs a 2D convolution over the input tensor with the provided weights and optional bias.
// Input shape: [batch, in_channels, in_h, in_w]
// Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
// Bias shape (optional): [out_channels]
func Conv2D(input, weight, bias *Tensor, strideH, strideW, padH, padW int) (*Tensor, error) {
	if len(input.shape) != 4 {
		return nil, errors.New("Conv2D expects input shape [batch, channels, height, width]")
	}
	if len(weight.shape) != 4 {
		return nil, errors.New("Conv2D expects weight shape [out_channels, in_channels, kernel_h, kernel_w]")
	}
	if bias != nil && len(bias.shape) != 1 {
		return nil, errors.New("bias for Conv2D must be rank 1")
	}

	batch := input.shape[0]
	inChannels := input.shape[1]
	inH := input.shape[2]
	inW := input.shape[3]
	outChannels := weight.shape[0]
	kernelChannels := weight.shape[1]
	kernelH := weight.shape[2]
	kernelW := weight.shape[3]

	if kernelChannels != inChannels {
		return nil, errors.New("kernel in_channels mismatch")
	}
	if strideH <= 0 || strideW <= 0 {
		return nil, errors.New("stride must be positive")
	}

	outH := (inH+2*padH-kernelH)/strideH + 1
	outW := (inW+2*padW-kernelW)/strideW + 1
	if outH <= 0 || outW <= 0 {
		return nil, errors.New("invalid output size")
	}

	out := Zeros(batch, outChannels, outH, outW)
	for n := 0; n < batch; n++ {
		for oc := 0; oc < outChannels; oc++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					acc := 0.0
					for ic := 0; ic < inChannels; ic++ {
						for kh := 0; kh < kernelH; kh++ {
							ih := oh*strideH - padH + kh
							if ih < 0 || ih >= inH {
								continue
							}
							for kw := 0; kw < kernelW; kw++ {
								iw := ow*strideW - padW + kw
								if iw < 0 || iw >= inW {
									continue
								}
								inputIdx := ((n*inChannels+ic)*inH+ih)*inW + iw
								weightIdx := ((oc*inChannels+ic)*kernelH+kh)*kernelW + kw
								acc += input.data[inputIdx] * weight.data[weightIdx]
							}
						}
					}
					if bias != nil {
						acc += bias.data[oc]
					}
					out.data[((n*outChannels+oc)*outH+oh)*outW+ow] = acc
				}
			}
		}
	}

	requiresGrad := input.requiresGrad || weight.requiresGrad || (bias != nil && bias.requiresGrad)
	if !requiresGrad {
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
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								gVal := grad.data[((n*outChannels+oc)*outH+oh)*outW+ow]
								for ic := 0; ic < inChannels; ic++ {
									for kh := 0; kh < kernelH; kh++ {
										ih := oh*strideH - padH + kh
										if ih < 0 || ih >= inH {
											continue
										}
										for kw := 0; kw < kernelW; kw++ {
											iw := ow*strideW - padW + kw
											if iw < 0 || iw >= inW {
												continue
											}
											inputIdx := ((n*inChannels+ic)*inH+ih)*inW + iw
											weightIdx := ((oc*inChannels+ic)*kernelH+kh)*kernelW + kw
											gInput.data[inputIdx] += weight.data[weightIdx] * gVal
										}
									}
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
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								gVal := grad.data[((n*outChannels+oc)*outH+oh)*outW+ow]
								for ic := 0; ic < inChannels; ic++ {
									for kh := 0; kh < kernelH; kh++ {
										ih := oh*strideH - padH + kh
										if ih < 0 || ih >= inH {
											continue
										}
										for kw := 0; kw < kernelW; kw++ {
											iw := ow*strideW - padW + kw
											if iw < 0 || iw >= inW {
												continue
											}
											inputIdx := ((n*inChannels+ic)*inH+ih)*inW + iw
											weightIdx := ((oc*inChannels+ic)*kernelH+kh)*kernelW + kw
											gWeight.data[weightIdx] += input.data[inputIdx] * gVal
										}
									}
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
						for oh := 0; oh < outH; oh++ {
							for ow := 0; ow < outW; ow++ {
								gBias.data[oc] += grad.data[((n*outChannels+oc)*outH+oh)*outW+ow]
							}
						}
					}
				}
				accumulate(grads, bias, gBias)
			}
		},
	}

	return out, nil
}
