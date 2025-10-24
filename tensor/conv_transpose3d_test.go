package tensor

import "testing"

func naiveConvTranspose3D(
	input, weight, bias []float64,
	batch, inChannels, inD, inH, inW int,
	outChannels, kernelD, kernelH, kernelW int,
	strideD, strideH, strideW, padD, padH, padW int,
) ([]float64, []int) {
	outD := (inD-1)*strideD - 2*padD + kernelD
	outH := (inH-1)*strideH - 2*padH + kernelH
	outW := (inW-1)*strideW - 2*padW + kernelW
	if outD <= 0 || outH <= 0 || outW <= 0 {
		return nil, nil
	}
	outData := make([]float64, batch*outChannels*outD*outH*outW)
	shape := []int{batch, outChannels, outD, outH, outW}
	for n := 0; n < batch; n++ {
		for ic := 0; ic < inChannels; ic++ {
			for id := 0; id < inD; id++ {
				for ih := 0; ih < inH; ih++ {
					for iw := 0; iw < inW; iw++ {
						inputVal := input[(((n*inChannels+ic)*inD+id)*inH+ih)*inW+iw]
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
										wIdx := ((((ic*outChannels)+oc)*kernelD+kd)*kernelH+kh)*kernelW + kw
										oIdx := ((((n*outChannels)+oc)*outD+od)*outH+oh)*outW + ow
										outData[oIdx] += inputVal * weight[wIdx]
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
				biasVal := bias[oc]
				if biasVal == 0 {
					continue
				}
				for od := 0; od < outD; od++ {
					for oh := 0; oh < outH; oh++ {
						for ow := 0; ow < outW; ow++ {
							oIdx := ((((n*outChannels)+oc)*outD+od)*outH+oh)*outW + ow
							outData[oIdx] += biasVal
						}
					}
				}
			}
		}
	}
	return outData, shape
}

func TestConvTranspose3DForwardBias(t *testing.T) {
	input := MustNew([]float64{1}, 1, 1, 1, 1, 1)
	weight := MustNew([]float64{1, 1, 1, 1, 1, 1, 1, 1}, 1, 1, 2, 2, 2)
	bias := MustNew([]float64{0.5}, 1)

	out, err := ConvTranspose3D(input, weight, bias, 1, 1, 1, 0, 0, 0)
	if err != nil {
		t.Fatalf("ConvTranspose3D returned error: %v", err)
	}

	expectedShape := []int{1, 1, 2, 2, 2}
	if len(out.shape) != len(expectedShape) {
		t.Fatalf("unexpected output rank: got %d want %d", len(out.shape), len(expectedShape))
	}
	for i, dim := range expectedShape {
		if out.shape[i] != dim {
			t.Fatalf("unexpected out.shape[%d]: got %d want %d", i, out.shape[i], dim)
		}
	}

	data := out.Data()
	if len(data) != 8 {
		t.Fatalf("unexpected output size: got %d want 8", len(data))
	}
	for i, v := range data {
		if v != 1.5 {
			t.Fatalf("unexpected output at %d: got %v want 1.5", i, v)
		}
	}
}

func TestConvTranspose3DBackward(t *testing.T) {
	input := MustNew([]float64{1}, 1, 1, 1, 1, 1)
	input.SetRequiresGrad(true)
	weight := MustNew([]float64{1, 1, 1, 1, 1, 1, 1, 1}, 1, 1, 2, 2, 2)
	weight.SetRequiresGrad(true)
	bias := MustNew([]float64{0}, 1)
	bias.SetRequiresGrad(true)

	out, err := ConvTranspose3D(input, weight, bias, 1, 1, 1, 0, 0, 0)
	if err != nil {
		t.Fatalf("ConvTranspose3D returned error: %v", err)
	}

	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	gInput := input.Grad()
	if gInput == nil {
		t.Fatalf("expected input gradient")
	}

	gInputData := gInput.Data()
	if len(gInputData) != 1 {
		t.Fatalf("unexpected input grad size: got %d want 1", len(gInputData))
	}
	if gInputData[0] != 8 {
		t.Fatalf("unexpected input grad value: got %v want 8", gInputData[0])
	}

	gWeight := weight.Grad()
	if gWeight == nil {
		t.Fatalf("expected weight gradient")
	}
	gWeightData := gWeight.Data()
	for i, v := range gWeightData {
		if v != 1 {
			t.Fatalf("unexpected weight grad at %d: got %v want 1", i, v)
		}
	}

	gBias := bias.Grad()
	if gBias == nil {
		t.Fatalf("expected bias gradient")
	}
	gBiasData := gBias.Data()
	if len(gBiasData) != 1 {
		t.Fatalf("unexpected bias grad size: got %d want 1", len(gBiasData))
	}
	if gBiasData[0] != 8 {
		t.Fatalf("unexpected bias grad value: got %v want 8", gBiasData[0])
	}
}

func TestConvTranspose3DForwardStridePadding(t *testing.T) {
	inputVals := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	input := MustNew(inputVals, 1, 1, 2, 2, 2)
	weightVals := []float64{0.5, -1, 1.5, 2, -0.5, 0.25, -1.25, 0.75}
	weight := MustNew(weightVals, 1, 1, 2, 2, 2)
	out, err := ConvTranspose3D(input, weight, nil, 2, 2, 2, 0, 0, 0)
	if err != nil {
		t.Fatalf("ConvTranspose3D returned error: %v", err)
	}
	expectedData, expectedShape := naiveConvTranspose3D(
		inputVals,
		weightVals,
		nil,
		1, 1, 2, 2, 2,
		1, 2, 2, 2,
		2, 2, 2, 0, 0, 0,
	)
	if expectedShape == nil {
		t.Fatalf("naiveConvTranspose3D produced invalid shape")
	}
	if len(out.shape) != len(expectedShape) {
		t.Fatalf("unexpected output rank: got %d want %d", len(out.shape), len(expectedShape))
	}
	for i, dim := range expectedShape {
		if out.shape[i] != dim {
			t.Fatalf("unexpected out.shape[%d]: got %d want %d", i, out.shape[i], dim)
		}
	}
	actual := out.Data()
	if len(actual) != len(expectedData) {
		t.Fatalf("unexpected data length: got %d want %d", len(actual), len(expectedData))
	}
	for i := range actual {
		if actual[i] != expectedData[i] {
			t.Fatalf("value mismatch at %d: got %v want %v", i, actual[i], expectedData[i])
		}
	}
}

func TestConvTranspose3DBackwardStridePadding(t *testing.T) {
	inputVals := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	input := MustNew(inputVals, 1, 1, 2, 2, 2)
	input.SetRequiresGrad(true)
	weightVals := []float64{0.5, -1, 1.5, 2, -0.5, 0.25, -1.25, 0.75}
	weight := MustNew(weightVals, 1, 1, 2, 2, 2)
	weight.SetRequiresGrad(true)
	out, err := ConvTranspose3D(input, weight, nil, 2, 2, 2, 0, 0, 0)
	if err != nil {
		t.Fatalf("ConvTranspose3D returned error: %v", err)
	}
	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	gInput := input.Grad()
	if gInput == nil {
		t.Fatalf("expected input gradient")
	}
	gInputData := gInput.Data()
	if len(gInputData) != len(inputVals) {
		t.Fatalf("unexpected input grad size: got %d want %d", len(gInputData), len(inputVals))
	}
	weightSum := 0.0
	for _, v := range weightVals {
		weightSum += v
	}
	for i, v := range gInputData {
		if v != weightSum {
			t.Fatalf("unexpected input grad at %d: got %v want %v", i, v, weightSum)
		}
	}
	gWeight := weight.Grad()
	if gWeight == nil {
		t.Fatalf("expected weight gradient")
	}
	gWeightData := gWeight.Data()
	if len(gWeightData) != len(weightVals) {
		t.Fatalf("unexpected weight grad size: got %d want %d", len(gWeightData), len(weightVals))
	}
	inputSum := 0.0
	for _, v := range inputVals {
		inputSum += v
	}
	for i, v := range gWeightData {
		if v != inputSum {
			t.Fatalf("unexpected weight grad at %d: got %v want %v", i, v, inputSum)
		}
	}
}

func TestConvTranspose3DForwardMultiChannel(t *testing.T) {
	inputVals := []float64{1, 2, 3, 4}
	input := MustNew(inputVals, 1, 2, 2, 1, 1)
	weightVals := []float64{0.5, 0.75, -0.25, 0.5, -1.0, 0.4, 1.2, -0.8}
	weight := MustNew(weightVals, 2, 2, 2, 1, 1)
	biasVals := []float64{0.1, -0.2}
	bias := MustNew(biasVals, 2)
	out, err := ConvTranspose3D(input, weight, bias, 1, 1, 1, 0, 0, 0)
	if err != nil {
		t.Fatalf("ConvTranspose3D returned error: %v", err)
	}
	expectedData, expectedShape := naiveConvTranspose3D(
		inputVals,
		weightVals,
		biasVals,
		1, 2, 2, 1, 1,
		2, 2, 1, 1,
		1, 1, 1, 0, 0, 0,
	)
	if expectedShape == nil {
		t.Fatalf("naiveConvTranspose3D produced invalid shape")
	}
	if len(out.shape) != len(expectedShape) {
		t.Fatalf("unexpected output rank: got %d want %d", len(out.shape), len(expectedShape))
	}
	for i, dim := range expectedShape {
		if out.shape[i] != dim {
			t.Fatalf("unexpected out.shape[%d]: got %d want %d", i, out.shape[i], dim)
		}
	}
	actual := out.Data()
	if len(actual) != len(expectedData) {
		t.Fatalf("unexpected output size: got %d want %d", len(actual), len(expectedData))
	}
	for i := range actual {
		if actual[i] != expectedData[i] {
			t.Fatalf("value mismatch at %d: got %v want %v", i, actual[i], expectedData[i])
		}
	}
}

func TestConvTranspose3DForwardPadding(t *testing.T) {
	inputVals := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	input := MustNew(inputVals, 1, 1, 2, 2, 2)
	weightVals := []float64{
		1, -1, 0.5,
		-0.25, 0.75, -0.6,
		0.8, -1.2, 0.3,
		-0.9, 1.1, -0.4,
		0.2, -0.5, 0.7,
		-1.5, 0.9, -0.2,
		0.6, -0.8, 0.1,
		-0.3, 0.4, -0.05,
		1.25, -0.65, 0.15,
	}
	weight := MustNew(weightVals, 1, 1, 3, 3, 3)
	out, err := ConvTranspose3D(input, weight, nil, 1, 1, 1, 1, 1, 1)
	if err != nil {
		t.Fatalf("ConvTranspose3D returned error: %v", err)
	}
	expectedData, expectedShape := naiveConvTranspose3D(
		inputVals,
		weightVals,
		nil,
		1, 1, 2, 2, 2,
		1, 3, 3, 3,
		1, 1, 1, 1, 1, 1,
	)
	if expectedShape == nil {
		t.Fatalf("naiveConvTranspose3D produced invalid shape")
	}
	for i, dim := range expectedShape {
		if out.shape[i] != dim {
			t.Fatalf("unexpected out.shape[%d]: got %d want %d", i, out.shape[i], dim)
		}
	}
	actual := out.Data()
	if len(actual) != len(expectedData) {
		t.Fatalf("unexpected output size: got %d want %d", len(actual), len(expectedData))
	}
	for i := range actual {
		if actual[i] != expectedData[i] {
			t.Fatalf("value mismatch at %d: got %v want %v", i, actual[i], expectedData[i])
		}
	}
}

func TestConvTranspose3DErrors(t *testing.T) {
	badInput := MustNew([]float64{1, 2}, 1, 2, 1)
	weight := MustNew([]float64{1}, 1, 1, 1, 1, 1)
	if _, err := ConvTranspose3D(badInput, weight, nil, 1, 1, 1, 0, 0, 0); err == nil {
		t.Fatalf("expected error for invalid input rank")
	}
	goodInput := MustNew([]float64{1}, 1, 1, 1, 1, 1)
	badWeight := MustNew([]float64{1, 2}, 2, 1, 1, 1, 1)
	if _, err := ConvTranspose3D(goodInput, badWeight, nil, 1, 1, 1, 0, 0, 0); err == nil {
		t.Fatalf("expected error for input/weight channel mismatch")
	}
	negativeStrideInput := MustNew([]float64{1}, 1, 1, 1, 1, 1)
	if _, err := ConvTranspose3D(negativeStrideInput, weight, nil, 0, 1, 1, 0, 0, 0); err == nil {
		t.Fatalf("expected error for non-positive stride")
	}
}
