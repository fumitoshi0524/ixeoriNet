package tensor

import "errors"

// MaxPool2D applies 2D max pooling on the input tensor.
// Input shape: [batch, channels, in_h, in_w]
func MaxPool2D(input *Tensor, kernelH, kernelW, strideH, strideW, padH, padW int) (*Tensor, error) {
	if len(input.shape) != 4 {
		return nil, errors.New("MaxPool2D expects input shape [batch, channels, height, width]")
	}
	if kernelH <= 0 || kernelW <= 0 {
		return nil, errors.New("kernel size must be positive")
	}
	if strideH <= 0 || strideW <= 0 {
		return nil, errors.New("stride must be positive")
	}
	batch := input.shape[0]
	channels := input.shape[1]
	inH := input.shape[2]
	inW := input.shape[3]
	outH := (inH+2*padH-kernelH)/strideH + 1
	outW := (inW+2*padW-kernelW)/strideW + 1
	if outH <= 0 || outW <= 0 {
		return nil, errors.New("invalid output size")
	}

	total := batch * channels * outH * outW
	indices := make([]int, total)
	out := Zeros(batch, channels, outH, outW)

	for n := 0; n < batch; n++ {
		for c := 0; c < channels; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					bestVal := -1.0
					bestIdx := -1
					first := true
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
							idx := ((n*channels+c)*inH+ih)*inW + iw
							val := input.data[idx]
							if first || val > bestVal {
								bestVal = val
								bestIdx = idx
								first = false
							}
						}
					}
					out.data[((n*channels+c)*outH+oh)*outW+ow] = bestVal
					indices[((n*channels+c)*outH+oh)*outW+ow] = bestIdx
				}
			}
		}
	}

	requiresGrad := input.requiresGrad
	if !requiresGrad {
		return out, nil
	}

	out.requiresGrad = true
	out.parents = []*Tensor{input}
	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			gInput := Zeros(input.shape...)
			for idx, src := range indices {
				if src < 0 {
					continue
				}
				gInput.data[src] += grad.data[idx]
			}
			accumulate(grads, input, gInput)
		},
	}

	return out, nil
}

// AvgPool2D applies 2D average pooling on the input tensor.
// Input shape: [batch, channels, in_h, in_w]
func AvgPool2D(input *Tensor, kernelH, kernelW, strideH, strideW, padH, padW int) (*Tensor, error) {
	if len(input.shape) != 4 {
		return nil, errors.New("AvgPool2D expects input shape [batch, channels, height, width]")
	}
	if kernelH <= 0 || kernelW <= 0 {
		return nil, errors.New("kernel size must be positive")
	}
	if strideH <= 0 || strideW <= 0 {
		return nil, errors.New("stride must be positive")
	}
	batch := input.shape[0]
	channels := input.shape[1]
	inH := input.shape[2]
	inW := input.shape[3]
	outH := (inH+2*padH-kernelH)/strideH + 1
	outW := (inW+2*padW-kernelW)/strideW + 1
	if outH <= 0 || outW <= 0 {
		return nil, errors.New("invalid output size")
	}

	out := Zeros(batch, channels, outH, outW)

	for n := 0; n < batch; n++ {
		for c := 0; c < channels; c++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					sum := 0.0
					count := 0
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
							idx := ((n*channels+c)*inH+ih)*inW + iw
							sum += input.data[idx]
							count++
						}
					}
					if count == 0 {
						return nil, errors.New("AvgPool2D kernel has no overlap with input")
					}
					out.data[((n*channels+c)*outH+oh)*outW+ow] = sum / float64(count)
				}
			}
		}
	}

	if !input.requiresGrad {
		return out, nil
	}

	out.requiresGrad = true
	out.parents = []*Tensor{input}
	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			gInput := Zeros(input.shape...)
			for n := 0; n < batch; n++ {
				for c := 0; c < channels; c++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							gVal := grad.data[((n*channels+c)*outH+oh)*outW+ow]
							count := 0
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
									count++
								}
							}
							if count == 0 {
								continue
							}
							share := gVal / float64(count)
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
									idx := ((n*channels+c)*inH+ih)*inW + iw
									gInput.data[idx] += share
								}
							}
						}
					}
				}
			}
			accumulate(grads, input, gInput)
		},
	}

	return out, nil
}
