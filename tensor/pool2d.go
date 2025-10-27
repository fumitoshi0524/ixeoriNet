package tensor

import (
	"errors"
	"math"
	"sync/atomic"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

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

	channelTotal := batch * channels
	parallel.For(channelTotal, func(start, end int) {
		for nc := start; nc < end; nc++ {
			n := nc / channels
			c := nc % channels
			outBase := (n*channels + c) * outH * outW
			inBase := (n*channels + c) * inH * inW
			for oh := 0; oh < outH; oh++ {
				ihBase := oh*strideH - padH
				outRow := outBase + oh*outW
				for ow := 0; ow < outW; ow++ {
					iwBase := ow*strideW - padW
					bestVal := math.Inf(-1)
					bestIdx := -1
					for kh := 0; kh < kernelH; kh++ {
						ih := ihBase + kh
						if ih < 0 || ih >= inH {
							continue
						}
						inputRow := inBase + ih*inW
						for kw := 0; kw < kernelW; kw++ {
							iw := iwBase + kw
							if iw < 0 || iw >= inW {
								continue
							}
							idx := inputRow + iw
							val := input.data[idx]
							if val > bestVal {
								bestVal = val
								bestIdx = idx
							}
						}
					}
					out.data[outRow+ow] = bestVal
					indices[outRow+ow] = bestIdx
				}
			}
		}
	})

	requiresGrad := input.requiresGrad
	if !requiresGrad {
		return out, nil
	}

	out.requiresGrad = true
	out.parents = []*Tensor{input}
	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			gInput := Zeros(input.shape...)
			parallel.For(batch, func(start, end int) {
				for n := start; n < end; n++ {
					gradBase := n * channels * outH * outW
					for c := 0; c < channels; c++ {
						offsetOut := gradBase + c*outH*outW
						offsetIdx := (n*channels + c) * outH * outW
						for i := 0; i < outH*outW; i++ {
							src := indices[offsetIdx+i]
							if src < 0 {
								continue
							}
							gInput.data[src] += grad.data[offsetOut+i]
						}
					}
				}
			})
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

	channelTotal := batch * channels
	var errFlag int32
	errValue := errors.New("AvgPool2D kernel has no overlap with input")
	parallel.For(channelTotal, func(start, end int) {
		for nc := start; nc < end; nc++ {
			if atomic.LoadInt32(&errFlag) == 1 {
				return
			}
			n := nc / channels
			c := nc % channels
			outBase := (n*channels + c) * outH * outW
			inBase := (n*channels + c) * inH * inW
			for oh := 0; oh < outH; oh++ {
				ihBase := oh*strideH - padH
				outRow := outBase + oh*outW
				for ow := 0; ow < outW; ow++ {
					iwBase := ow*strideW - padW
					sum := 0.0
					count := 0
					for kh := 0; kh < kernelH; kh++ {
						ih := ihBase + kh
						if ih < 0 || ih >= inH {
							continue
						}
						inputRow := inBase + ih*inW
						for kw := 0; kw < kernelW; kw++ {
							iw := iwBase + kw
							if iw < 0 || iw >= inW {
								continue
							}
							idx := inputRow + iw
							sum += input.data[idx]
							count++
						}
					}
					if count == 0 {
						atomic.StoreInt32(&errFlag, 1)
						return
					}
					out.data[outRow+ow] = sum / float64(count)
				}
			}
		}
	})
	if atomic.LoadInt32(&errFlag) == 1 {
		return nil, errValue
	}

	if !input.requiresGrad {
		return out, nil
	}

	out.requiresGrad = true
	out.parents = []*Tensor{input}
	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			gInput := Zeros(input.shape...)
			parallel.For(batch, func(start, end int) {
				for n := start; n < end; n++ {
					inBase := n * channels * inH * inW
					gradBase := n * channels * outH * outW
					for c := 0; c < channels; c++ {
						gradOffset := gradBase + c*outH*outW
						for oh := 0; oh < outH; oh++ {
							ihBase := oh*strideH - padH
							gradRow := gradOffset + oh*outW
							for ow := 0; ow < outW; ow++ {
								iwBase := ow*strideW - padW
								gVal := grad.data[gradRow+ow]
								if gVal == 0 {
									continue
								}
								count := 0
								for kh := 0; kh < kernelH; kh++ {
									ih := ihBase + kh
									if ih < 0 || ih >= inH {
										continue
									}
									for kw := 0; kw < kernelW; kw++ {
										iw := iwBase + kw
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
								inputChannelOffset := inBase + c*inH*inW
								for kh := 0; kh < kernelH; kh++ {
									ih := ihBase + kh
									if ih < 0 || ih >= inH {
										continue
									}
									inputRow := inputChannelOffset + ih*inW
									for kw := 0; kw < kernelW; kw++ {
										iw := iwBase + kw
										if iw < 0 || iw >= inW {
											continue
										}
										idx := inputRow + iw
										gInput.data[idx] += share
									}
								}
							}
						}
					}
				}
			})
			accumulate(grads, input, gInput)
		},
	}

	return out, nil
}
