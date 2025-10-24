package tensor

import "errors"

// ConvTranspose3D performs a 3D transposed convolution (deconvolution).
// Input shape: [batch, in_channels, in_d, in_h, in_w]
// Weight shape: [in_channels, out_channels, kernel_d, kernel_h, kernel_w]
func ConvTranspose3D(input, weight, bias *Tensor, strideD, strideH, strideW, padD, padH, padW int) (*Tensor, error) {
	if len(input.shape) != 5 {
		return nil, errors.New("ConvTranspose3D expects input shape [batch, channels, depth, height, width]")
	}
	if len(weight.shape) != 5 {
		return nil, errors.New("ConvTranspose3D expects weight shape [in_channels, out_channels, kernel_d, kernel_h, kernel_w]")
	}
	if bias != nil && len(bias.shape) != 1 {
		return nil, errors.New("bias for ConvTranspose3D must be rank 1")
	}

	batch := input.shape[0]
	inChannels := input.shape[1]
	inD := input.shape[2]
	inH := input.shape[3]
	inW := input.shape[4]
	weightInChannels := weight.shape[0]
	outChannels := weight.shape[1]
	kernelD := weight.shape[2]
	kernelH := weight.shape[3]
	kernelW := weight.shape[4]

	if weightInChannels != inChannels {
		return nil, errors.New("weight in_channels mismatch")
	}
	if strideD <= 0 || strideH <= 0 || strideW <= 0 {
		return nil, errors.New("stride must be positive")
	}

	outD := (inD-1)*strideD - 2*padD + kernelD
	outH := (inH-1)*strideH - 2*padH + kernelH
	outW := (inW-1)*strideW - 2*padW + kernelW
	if outD <= 0 || outH <= 0 || outW <= 0 {
		return nil, errors.New("invalid output size")
	}

	out := Zeros(batch, outChannels, outD, outH, outW)
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for id := 0; id < inD; id++ {
				for ih := 0; ih < inH; ih++ {
					for iw := 0; iw < inW; iw++ {
						inputVal := input.data[(((n*inChannels+ic)*inD+id)*inH+ih)*inW+iw]
						if inputVal == 0 {
							continue
						}
						for oc := 0; oc < outChannels; oc++ {
							for kd := 0; kd < kernelD; kd++ {
								od := id*strideD - padD + kd
								if od < 0 || od >= outD {
									continue
								}
								for kh := 0; kh < kernelH; kh++ {
									oh := ih*strideH - padH + kh
									if oh < 0 || oh >= outH {
										continue
									}
									for kw := 0; kw < kernelW; kw++ {
										ow := iw*strideW - padW + kw
										if ow < 0 || ow >= outW {
											continue
										}
										weightIdx := ((((ic*outChannels)+oc)*kernelD+kd)*kernelH+kh)*kernelW + kw
										outIdx := ((((n*outChannels)+oc)*outD+od)*outH+oh)*outW + ow
										out.data[outIdx] += inputVal * weight.data[weightIdx]
									}
								}
							}
						}
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
				for od := 0; od < outD; od++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							outIdx := ((((n*outChannels)+oc)*outD+od)*outH+oh)*outW + ow
							out.data[outIdx] += biasVal
						}
					}
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
						for id := 0; id < inD; id++ {
							for ih := 0; ih < inH; ih++ {
								for iw := 0; iw < inW; iw++ {
									sum := 0.0
									for oc := 0; oc < outChannels; oc++ {
										for kd := 0; kd < kernelD; kd++ {
											od := id*strideD - padD + kd
											if od < 0 || od >= outD {
												continue
											}
											for kh := 0; kh < kernelH; kh++ {
												oh := ih*strideH - padH + kh
												if oh < 0 || oh >= outH {
													continue
												}
												for kw := 0; kw < kernelW; kw++ {
													ow := iw*strideW - padW + kw
													if ow < 0 || ow >= outW {
														continue
													}
													gradIdx := ((((n*outChannels)+oc)*outD+od)*outH+oh)*outW + ow
													weightIdx := ((((ic*outChannels)+oc)*kernelD+kd)*kernelH+kh)*kernelW + kw
													sum += grad.data[gradIdx] * weight.data[weightIdx]
												}
											}
										}
									}
									inputIdx := (((n*inChannels+ic)*inD+id)*inH+ih)*inW + iw
									gInput.data[inputIdx] = sum
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
					for ic := 0; ic < inChannels; ic++ {
						for id := 0; id < inD; id++ {
							for ih := 0; ih < inH; ih++ {
								for iw := 0; iw < inW; iw++ {
									inputVal := input.data[(((n*inChannels+ic)*inD+id)*inH+ih)*inW+iw]
									if inputVal == 0 {
										continue
									}
									for oc := 0; oc < outChannels; oc++ {
										for kd := 0; kd < kernelD; kd++ {
											od := id*strideD - padD + kd
											if od < 0 || od >= outD {
												continue
											}
											for kh := 0; kh < kernelH; kh++ {
												oh := ih*strideH - padH + kh
												if oh < 0 || oh >= outH {
													continue
												}
												for kw := 0; kw < kernelW; kw++ {
													ow := iw*strideW - padW + kw
													if ow < 0 || ow >= outW {
														continue
													}
													gradIdx := ((((n*outChannels)+oc)*outD+od)*outH+oh)*outW + ow
													weightIdx := ((((ic*outChannels)+oc)*kernelD+kd)*kernelH+kh)*kernelW + kw
													gWeight.data[weightIdx] += grad.data[gradIdx] * inputVal
												}
											}
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
						for od := 0; od < outD; od++ {
							for oh := 0; oh < outH; oh++ {
								for ow := 0; ow < outW; ow++ {
									gradIdx := ((((n*outChannels)+oc)*outD+od)*outH+oh)*outW + ow
									gBias.data[oc] += grad.data[gradIdx]
								}
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
