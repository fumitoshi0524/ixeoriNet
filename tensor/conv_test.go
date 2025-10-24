package tensor

import "testing"

func naiveConv1DForward(input, weight, bias []float64, batch, inChannels, inW, outChannels, kernel, stride, pad int) []float64 {
	outW := (inW+2*pad-kernel)/stride + 1
	out := make([]float64, batch*outChannels*outW)
	for n := 0; n < batch; n++ {
		for oc := 0; oc < outChannels; oc++ {
			for ow := 0; ow < outW; ow++ {
				acc := 0.0
				for ic := 0; ic < inChannels; ic++ {
					for kw := 0; kw < kernel; kw++ {
						iw := ow*stride - pad + kw
						if iw < 0 || iw >= inW {
							continue
						}
						inputIdx := ((n*inChannels+ic)*inW + iw)
						weightIdx := ((oc*inChannels+ic)*kernel + kw)
						acc += input[inputIdx] * weight[weightIdx]
					}
				}
				if bias != nil {
					acc += bias[oc]
				}
				out[(n*outChannels+oc)*outW+ow] = acc
			}
		}
	}
	return out
}

func naiveConv1DInputGrad(input, weight []float64, batch, inChannels, inW, outChannels, kernel, stride, pad int, gradOut []float64) []float64 {
	gInput := make([]float64, len(input))
	outW := (inW+2*pad-kernel)/stride + 1
	for n := 0; n < batch; n++ {
		for oc := 0; oc < outChannels; oc++ {
			for ow := 0; ow < outW; ow++ {
				gVal := gradOut[(n*outChannels+oc)*outW+ow]
				for ic := 0; ic < inChannels; ic++ {
					for kw := 0; kw < kernel; kw++ {
						iw := ow*stride - pad + kw
						if iw < 0 || iw >= inW {
							continue
						}
						inputIdx := ((n*inChannels+ic)*inW + iw)
						weightIdx := ((oc*inChannels+ic)*kernel + kw)
						gInput[inputIdx] += weight[weightIdx] * gVal
					}
				}
			}
		}
	}
	return gInput
}

func naiveConv1DWeightGrad(input, gradOut []float64, batch, inChannels, inW, outChannels, kernel, stride, pad int) []float64 {
	gWeight := make([]float64, outChannels*inChannels*kernel)
	outW := (inW+2*pad-kernel)/stride + 1
	for n := 0; n < batch; n++ {
		for oc := 0; oc < outChannels; oc++ {
			for ow := 0; ow < outW; ow++ {
				gVal := gradOut[(n*outChannels+oc)*outW+ow]
				for ic := 0; ic < inChannels; ic++ {
					for kw := 0; kw < kernel; kw++ {
						iw := ow*stride - pad + kw
						if iw < 0 || iw >= inW {
							continue
						}
						inputIdx := ((n*inChannels+ic)*inW + iw)
						weightIdx := ((oc*inChannels+ic)*kernel + kw)
						gWeight[weightIdx] += input[inputIdx] * gVal
					}
				}
			}
		}
	}
	return gWeight
}

func almostEqualSlices(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}

func TestConv1DForwardBackward(t *testing.T) {
	inputVals := []float64{1, 2, 3, 4}
	weightVals := []float64{0.5, -1.0, 0.25, 1.5}
	biasVals := []float64{0.2}
	input := MustNew(inputVals, 1, 1, 4)
	input.SetRequiresGrad(true)
	weight := MustNew(weightVals, 1, 1, 4)
	weight.SetRequiresGrad(true)
	bias := MustNew(biasVals, 1)
	bias.SetRequiresGrad(true)

	out, err := Conv1D(input, weight, bias, 1, 1)
	if err != nil {
		t.Fatalf("Conv1D returned error: %v", err)
	}

	expected := naiveConv1DForward(inputVals, weightVals, biasVals, 1, 1, 4, 1, 4, 1, 1)
	if !almostEqualSlices(out.data, expected, 1e-9) {
		t.Fatalf("Conv1D forward mismatch: got %v want %v", out.data, expected)
	}

	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	gradOut := make([]float64, len(out.data))
	for i := range gradOut {
		gradOut[i] = 1
	}
	expectedInputGrad := naiveConv1DInputGrad(inputVals, weightVals, 1, 1, 4, 1, 4, 1, 1, gradOut)
	gInput := input.Grad()
	if gInput == nil || !almostEqualSlices(gInput.data, expectedInputGrad, 1e-9) {
		t.Fatalf("Conv1D input grad mismatch: got %v want %v", gInput.data, expectedInputGrad)
	}

	expectedWeightGrad := naiveConv1DWeightGrad(inputVals, gradOut, 1, 1, 4, 1, 4, 1, 1)
	gWeight := weight.Grad()
	if gWeight == nil || !almostEqualSlices(gWeight.data, expectedWeightGrad, 1e-9) {
		t.Fatalf("Conv1D weight grad mismatch: got %v want %v", gWeight.data, expectedWeightGrad)
	}

	gBias := bias.Grad()
	if gBias == nil || !almostEqualSlices(gBias.data, []float64{float64(len(gradOut))}, 1e-9) {
		t.Fatalf("Conv1D bias grad mismatch: got %v want %v", gBias.data, []float64{float64(len(gradOut))})
	}
}

func naiveConv2DForward(input, weight, bias []float64, batch, inChannels, inH, inW, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int) []float64 {
	outH := (inH+2*padH-kernelH)/strideH + 1
	outW := (inW+2*padW-kernelW)/strideW + 1
	out := make([]float64, batch*outChannels*outH*outW)
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
								acc += input[inputIdx] * weight[weightIdx]
							}
						}
					}
					if bias != nil {
						acc += bias[oc]
					}
					out[((n*outChannels+oc)*outH+oh)*outW+ow] = acc
				}
			}
		}
	}
	return out
}

func naiveConv2DInputGrad(input, weight []float64, batch, inChannels, inH, inW, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int, gradOut []float64) []float64 {
	gInput := make([]float64, len(input))
	outH := (inH+2*padH-kernelH)/strideH + 1
	outW := (inW+2*padW-kernelW)/strideW + 1
	for n := 0; n < batch; n++ {
		for oc := 0; oc < outChannels; oc++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					gVal := gradOut[((n*outChannels+oc)*outH+oh)*outW+ow]
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
								gInput[inputIdx] += weight[weightIdx] * gVal
							}
						}
					}
				}
			}
		}
	}
	return gInput
}

func naiveConv2DWeightGrad(input, gradOut []float64, batch, inChannels, inH, inW, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int) []float64 {
	gWeight := make([]float64, outChannels*inChannels*kernelH*kernelW)
	outH := (inH+2*padH-kernelH)/strideH + 1
	outW := (inW+2*padW-kernelW)/strideW + 1
	for n := 0; n < batch; n++ {
		for oc := 0; oc < outChannels; oc++ {
			for oh := 0; oh < outH; oh++ {
				for ow := 0; ow < outW; ow++ {
					gVal := gradOut[((n*outChannels+oc)*outH+oh)*outW+ow]
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
								gWeight[weightIdx] += input[inputIdx] * gVal
							}
						}
					}
				}
			}
		}
	}
	return gWeight
}

func TestConv2DForwardBackward(t *testing.T) {
	inputVals := []float64{
		1, 2,
		3, 4,
	}
	weightVals := []float64{
		0.5, -1,
		2, 0.25,
	}
	biasVals := []float64{0.3}
	input := MustNew(inputVals, 1, 1, 2, 2)
	input.SetRequiresGrad(true)
	weight := MustNew(weightVals, 1, 1, 2, 2)
	weight.SetRequiresGrad(true)
	bias := MustNew(biasVals, 1)
	bias.SetRequiresGrad(true)
	out, err := Conv2D(input, weight, bias, 1, 1, 0, 0)
	if err != nil {
		t.Fatalf("Conv2D returned error: %v", err)
	}
	expected := naiveConv2DForward(inputVals, weightVals, biasVals, 1, 1, 2, 2, 1, 2, 2, 1, 1, 0, 0)
	if !almostEqualSlices(out.data, expected, 1e-9) {
		t.Fatalf("Conv2D forward mismatch: got %v want %v", out.data, expected)
	}
	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	gradOut := make([]float64, len(out.data))
	for i := range gradOut {
		gradOut[i] = 1
	}
	expectedInputGrad := naiveConv2DInputGrad(inputVals, weightVals, 1, 1, 2, 2, 1, 2, 2, 1, 1, 0, 0, gradOut)
	gInput := input.Grad()
	if gInput == nil || !almostEqualSlices(gInput.data, expectedInputGrad, 1e-9) {
		t.Fatalf("Conv2D input grad mismatch: got %v want %v", gInput.data, expectedInputGrad)
	}
	expectedWeightGrad := naiveConv2DWeightGrad(inputVals, gradOut, 1, 1, 2, 2, 1, 2, 2, 1, 1, 0, 0)
	gWeight := weight.Grad()
	if gWeight == nil || !almostEqualSlices(gWeight.data, expectedWeightGrad, 1e-9) {
		t.Fatalf("Conv2D weight grad mismatch: got %v want %v", gWeight.data, expectedWeightGrad)
	}
	gBias := bias.Grad()
	if gBias == nil || !almostEqualSlices(gBias.data, []float64{float64(len(gradOut))}, 1e-9) {
		t.Fatalf("Conv2D bias grad mismatch: got %v want %v", gBias.data, []float64{float64(len(gradOut))})
	}
}

func naiveConvTranspose1DForward(input, weight, bias []float64, batch, inChannels, inW, outChannels, kernel, stride, pad int) []float64 {
	outW := (inW-1)*stride - 2*pad + kernel
	out := make([]float64, batch*outChannels*outW)
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for iw := 0; iw < inW; iw++ {
				val := input[(n*inChannels+ic)*inW+iw]
				if val == 0 {
					continue
				}
				for oc := 0; oc < outChannels; oc++ {
					for k := 0; k < kernel; k++ {
						ow := iw*stride - pad + k
						if ow < 0 || ow >= outW {
							continue
						}
						wIdx := (ic*outChannels+oc)*kernel + k
						oIdx := (n*outChannels+oc)*outW + ow
						out[oIdx] += val * weight[wIdx]
					}
				}
			}
		}
	}
	if bias != nil {
		for n := 0; n < batch; n++ {
			for oc := 0; oc < outChannels; oc++ {
				b := bias[oc]
				if b == 0 {
					continue
				}
				for ow := 0; ow < outW; ow++ {
					oIdx := (n*outChannels+oc)*outW + ow
					out[oIdx] += b
				}
			}
		}
	}
	return out
}

func naiveConvTranspose1DInputGrad(input, weight []float64, batch, inChannels, inW, outChannels, kernel, stride, pad int, gradOut []float64) []float64 {
	gInput := make([]float64, len(input))
	outW := (inW-1)*stride - 2*pad + kernel
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for iw := 0; iw < inW; iw++ {
				sum := 0.0
				base := iw*stride - pad
				for oc := 0; oc < outChannels; oc++ {
					for k := 0; k < kernel; k++ {
						ow := base + k
						if ow < 0 || ow >= outW {
							continue
						}
						gIdx := (n*outChannels+oc)*outW + ow
						wIdx := (ic*outChannels+oc)*kernel + k
						sum += gradOut[gIdx] * weight[wIdx]
					}
				}
				gInput[(n*inChannels+ic)*inW+iw] = sum
			}
		}
	}
	return gInput
}

func naiveConvTranspose1DWeightGrad(input, gradOut []float64, batch, inChannels, inW, outChannels, kernel, stride, pad int) []float64 {
	gWeight := make([]float64, inChannels*outChannels*kernel)
	outW := (inW-1)*stride - 2*pad + kernel
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for iw := 0; iw < inW; iw++ {
				val := input[(n*inChannels+ic)*inW+iw]
				if val == 0 {
					continue
				}
				base := iw*stride - pad
				for oc := 0; oc < outChannels; oc++ {
					for k := 0; k < kernel; k++ {
						ow := base + k
						if ow < 0 || ow >= outW {
							continue
						}
						gIdx := (n*outChannels+oc)*outW + ow
						wIdx := (ic*outChannels+oc)*kernel + k
						gWeight[wIdx] += gradOut[gIdx] * val
					}
				}
			}
		}
	}
	return gWeight
}

func TestConvTranspose1DForwardBackward(t *testing.T) {
	inputVals := []float64{1, -2}
	weightVals := []float64{0.5, -1, 1.5, 0.75}
	biasVals := []float64{0.1}
	input := MustNew(inputVals, 1, 1, 2)
	input.SetRequiresGrad(true)
	weight := MustNew(weightVals, 1, 1, 4)
	weight.SetRequiresGrad(true)
	bias := MustNew(biasVals, 1)
	bias.SetRequiresGrad(true)
	out, err := ConvTranspose1D(input, weight, bias, 2, 1)
	if err != nil {
		t.Fatalf("ConvTranspose1D returned error: %v", err)
	}
	expected := naiveConvTranspose1DForward(inputVals, weightVals, biasVals, 1, 1, 2, 1, 4, 2, 1)
	if !almostEqualSlices(out.data, expected, 1e-9) {
		t.Fatalf("ConvTranspose1D forward mismatch: got %v want %v", out.data, expected)
	}
	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	gradOut := make([]float64, len(out.data))
	for i := range gradOut {
		gradOut[i] = 1
	}
	expectedInputGrad := naiveConvTranspose1DInputGrad(inputVals, weightVals, 1, 1, 2, 1, 4, 2, 1, gradOut)
	gInput := input.Grad()
	if gInput == nil || !almostEqualSlices(gInput.data, expectedInputGrad, 1e-9) {
		t.Fatalf("ConvTranspose1D input grad mismatch: got %v want %v", gInput.data, expectedInputGrad)
	}
	expectedWeightGrad := naiveConvTranspose1DWeightGrad(inputVals, gradOut, 1, 1, 2, 1, 4, 2, 1)
	gWeight := weight.Grad()
	if gWeight == nil || !almostEqualSlices(gWeight.data, expectedWeightGrad, 1e-9) {
		t.Fatalf("ConvTranspose1D weight grad mismatch: got %v want %v", gWeight.data, expectedWeightGrad)
	}
	gBias := bias.Grad()
	if gBias == nil || !almostEqualSlices(gBias.data, []float64{float64(len(gradOut))}, 1e-9) {
		t.Fatalf("ConvTranspose1D bias grad mismatch: got %v want %v", gBias.data, []float64{float64(len(gradOut))})
	}
}

func naiveConvTranspose2DForward(input, weight, bias []float64, batch, inChannels, inH, inW, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int) []float64 {
	outH := (inH-1)*strideH - 2*padH + kernelH
	outW := (inW-1)*strideW - 2*padW + kernelW
	out := make([]float64, batch*outChannels*outH*outW)
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for ih := 0; ih < inH; ih++ {
				for iw := 0; iw < inW; iw++ {
					val := input[((n*inChannels+ic)*inH+ih)*inW+iw]
					if val == 0 {
						continue
					}
					for oc := 0; oc < outChannels; oc++ {
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
								wIdx := ((ic*outChannels+oc)*kernelH+kh)*kernelW + kw
								oIdx := ((n*outChannels+oc)*outH+oh)*outW + ow
								out[oIdx] += val * weight[wIdx]
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
				b := bias[oc]
				if b == 0 {
					continue
				}
				for oh := 0; oh < outH; oh++ {
					for ow := 0; ow < outW; ow++ {
						oIdx := ((n*outChannels+oc)*outH+oh)*outW + ow
						out[oIdx] += b
					}
				}
			}
		}
	}
	return out
}

func naiveConvTranspose2DInputGrad(input, weight []float64, batch, inChannels, inH, inW, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int, gradOut []float64) []float64 {
	gInput := make([]float64, len(input))
	outH := (inH-1)*strideH - 2*padH + kernelH
	outW := (inW-1)*strideW - 2*padW + kernelW
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for ih := 0; ih < inH; ih++ {
				for iw := 0; iw < inW; iw++ {
					sum := 0.0
					baseH := ih*strideH - padH
					baseW := iw*strideW - padW
					for oc := 0; oc < outChannels; oc++ {
						for kh := 0; kh < kernelH; kh++ {
							oh := baseH + kh
							if oh < 0 || oh >= outH {
								continue
							}
							for kw := 0; kw < kernelW; kw++ {
								ow := baseW + kw
								if ow < 0 || ow >= outW {
									continue
								}
								gIdx := ((n*outChannels+oc)*outH+oh)*outW + ow
								wIdx := ((ic*outChannels+oc)*kernelH+kh)*kernelW + kw
								sum += gradOut[gIdx] * weight[wIdx]
							}
						}
					}
					gInput[((n*inChannels+ic)*inH+ih)*inW+iw] = sum
				}
			}
		}
	}
	return gInput
}

func naiveConvTranspose2DWeightGrad(input, gradOut []float64, batch, inChannels, inH, inW, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int) []float64 {
	gWeight := make([]float64, inChannels*outChannels*kernelH*kernelW)
	outH := (inH-1)*strideH - 2*padH + kernelH
	outW := (inW-1)*strideW - 2*padW + kernelW
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for ih := 0; ih < inH; ih++ {
				for iw := 0; iw < inW; iw++ {
					val := input[((n*inChannels+ic)*inH+ih)*inW+iw]
					if val == 0 {
						continue
					}
					baseH := ih*strideH - padH
					baseW := iw*strideW - padW
					for oc := 0; oc < outChannels; oc++ {
						for kh := 0; kh < kernelH; kh++ {
							oh := baseH + kh
							if oh < 0 || oh >= outH {
								continue
							}
							for kw := 0; kw < kernelW; kw++ {
								ow := baseW + kw
								if ow < 0 || ow >= outW {
									continue
								}
								gIdx := ((n*outChannels+oc)*outH+oh)*outW + ow
								wIdx := ((ic*outChannels+oc)*kernelH+kh)*kernelW + kw
								gWeight[wIdx] += gradOut[gIdx] * val
							}
						}
					}
				}
			}
		}
	}
	return gWeight
}

func TestConvTranspose2DForwardBackward(t *testing.T) {
	inputVals := []float64{1, 2, 3, 4}
	weightVals := []float64{
		0.5, -1,
		0.75, 1.2,
	}
	biasVals := []float64{0.2}
	input := MustNew(inputVals, 1, 1, 2, 2)
	input.SetRequiresGrad(true)
	weight := MustNew(weightVals, 1, 1, 2, 2)
	weight.SetRequiresGrad(true)
	bias := MustNew(biasVals, 1)
	bias.SetRequiresGrad(true)
	out, err := ConvTranspose2D(input, weight, bias, 2, 2, 0, 0)
	if err != nil {
		t.Fatalf("ConvTranspose2D returned error: %v", err)
	}
	expected := naiveConvTranspose2DForward(inputVals, weightVals, biasVals, 1, 1, 2, 2, 1, 2, 2, 2, 2, 0, 0)
	if !almostEqualSlices(out.data, expected, 1e-9) {
		t.Fatalf("ConvTranspose2D forward mismatch: got %v want %v", out.data, expected)
	}
	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	gradOut := make([]float64, len(out.data))
	for i := range gradOut {
		gradOut[i] = 1
	}
	expectedInputGrad := naiveConvTranspose2DInputGrad(inputVals, weightVals, 1, 1, 2, 2, 1, 2, 2, 2, 2, 0, 0, gradOut)
	gInput := input.Grad()
	if gInput == nil || !almostEqualSlices(gInput.data, expectedInputGrad, 1e-9) {
		t.Fatalf("ConvTranspose2D input grad mismatch: got %v want %v", gInput.data, expectedInputGrad)
	}
	expectedWeightGrad := naiveConvTranspose2DWeightGrad(inputVals, gradOut, 1, 1, 2, 2, 1, 2, 2, 2, 2, 0, 0)
	gWeight := weight.Grad()
	if gWeight == nil || !almostEqualSlices(gWeight.data, expectedWeightGrad, 1e-9) {
		t.Fatalf("ConvTranspose2D weight grad mismatch: got %v want %v", gWeight.data, expectedWeightGrad)
	}
	gBias := bias.Grad()
	if gBias == nil || !almostEqualSlices(gBias.data, []float64{float64(len(gradOut))}, 1e-9) {
		t.Fatalf("ConvTranspose2D bias grad mismatch: got %v want %v", gBias.data, []float64{float64(len(gradOut))})
	}
}
