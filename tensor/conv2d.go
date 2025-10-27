package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

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
	totalChannels := batch * outChannels
	kernelArea := kernelH * kernelW
	inputHW := inH * inW
	outHW := outH * outW
	parallel.For(totalChannels, func(start, end int) {
		for noc := start; noc < end; noc++ {
			n := noc / outChannels
			oc := noc % outChannels
			batchOffset := n * inChannels * inputHW
			outBase := (n*outChannels + oc) * outHW
			weightOcOffset := oc * inChannels * kernelArea
			for oh := 0; oh < outH; oh++ {
				ihBase := oh*strideH - padH
				outRow := outBase + oh*outW
				for ow := 0; ow < outW; ow++ {
					iwBase := ow*strideW - padW
					acc := 0.0
					for ic := 0; ic < inChannels; ic++ {
						inputChannelOffset := batchOffset + ic*inputHW
						weightChannelOffset := weightOcOffset + ic*kernelArea
						for kh := 0; kh < kernelH; kh++ {
							ih := ihBase + kh
							if ih < 0 || ih >= inH {
								continue
							}
							inputRow := inputChannelOffset + ih*inW
							weightRow := weightChannelOffset + kh*kernelW
							for kw := 0; kw < kernelW; kw++ {
								iw := iwBase + kw
								if iw < 0 || iw >= inW {
									continue
								}
								inputIdx := inputRow + iw
								weightIdx := weightRow + kw
								acc += input.data[inputIdx] * weight.data[weightIdx]
							}
						}
					}
					if bias != nil {
						acc += bias.data[oc]
					}
					out.data[outRow+ow] = acc
				}
			}
		}
	})

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
				parallel.For(batch, func(start, end int) {
					for n := start; n < end; n++ {
						inBatchOffset := n * inChannels * inputHW
						gradBatchOffset := n * outChannels * outHW
						for oc := 0; oc < outChannels; oc++ {
							weightOcOffset := oc * inChannels * kernelArea
							gradChannelOffset := gradBatchOffset + oc*outHW
							for oh := 0; oh < outH; oh++ {
								ihBase := oh*strideH - padH
								gradRow := gradChannelOffset + oh*outW
								for ow := 0; ow < outW; ow++ {
									iwBase := ow*strideW - padW
									gVal := grad.data[gradRow+ow]
									if gVal == 0 {
										continue
									}
									for ic := 0; ic < inChannels; ic++ {
										inputChannelOffset := inBatchOffset + ic*inputHW
										weightChannelOffset := weightOcOffset + ic*kernelArea
										for kh := 0; kh < kernelH; kh++ {
											ih := ihBase + kh
											if ih < 0 || ih >= inH {
												continue
											}
											inputRow := inputChannelOffset + ih*inW
											weightRow := weightChannelOffset + kh*kernelW
											for kw := 0; kw < kernelW; kw++ {
												iw := iwBase + kw
												if iw < 0 || iw >= inW {
													continue
												}
												inputIdx := inputRow + iw
												weightIdx := weightRow + kw
												gInput.data[inputIdx] += weight.data[weightIdx] * gVal
											}
										}
									}
								}
							}
						}
					}
				})
				accumulate(grads, input, gInput)
			}

			if weight.requiresGrad {
				gWeight := Zeros(weight.shape...)
				parallel.For(outChannels, func(start, end int) {
					for oc := start; oc < end; oc++ {
						weightOcOffset := oc * inChannels * kernelArea
						for n := 0; n < batch; n++ {
							gradChannelOffset := ((n*outChannels + oc) * outH) * outW
							inBatchOffset := n * inChannels * inputHW
							for oh := 0; oh < outH; oh++ {
								ihBase := oh*strideH - padH
								gradRow := gradChannelOffset + oh*outW
								for ow := 0; ow < outW; ow++ {
									iwBase := ow*strideW - padW
									gVal := grad.data[gradRow+ow]
									if gVal == 0 {
										continue
									}
									for ic := 0; ic < inChannels; ic++ {
										inputChannelOffset := inBatchOffset + ic*inputHW
										weightChannelOffset := weightOcOffset + ic*kernelArea
										for kh := 0; kh < kernelH; kh++ {
											ih := ihBase + kh
											if ih < 0 || ih >= inH {
												continue
											}
											inputRow := inputChannelOffset + ih*inW
											weightRow := weightChannelOffset + kh*kernelW
											for kw := 0; kw < kernelW; kw++ {
												iw := iwBase + kw
												if iw < 0 || iw >= inW {
													continue
												}
												inputIdx := inputRow + iw
												weightIdx := weightRow + kw
												gWeight.data[weightIdx] += input.data[inputIdx] * gVal
											}
										}
									}
								}
							}
						}
					}
				})
				accumulate(grads, weight, gWeight)
			}

			if bias != nil && bias.requiresGrad {
				gBias := Zeros(bias.shape...)
				parallel.For(outChannels, func(start, end int) {
					for oc := start; oc < end; oc++ {
						sum := 0.0
						for n := 0; n < batch; n++ {
							gradChannelOffset := ((n*outChannels + oc) * outH) * outW
							for idx := 0; idx < outHW; idx++ {
								sum += grad.data[gradChannelOffset+idx]
							}
						}
						gBias.data[oc] = sum
					}
				})
				accumulate(grads, bias, gBias)
			}
		},
	}

	return out, nil
}
