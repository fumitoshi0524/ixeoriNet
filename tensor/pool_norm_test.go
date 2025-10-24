package tensor

import (
	"math"
	"testing"
)

func numericalGrad(f func([]float64) float64, base []float64, eps float64) []float64 {
	grad := make([]float64, len(base))
	for i := range base {
		plus := append([]float64(nil), base...)
		plus[i] += eps
		fPlus := f(plus)
		minus := append([]float64(nil), base...)
		minus[i] -= eps
		fMinus := f(minus)
		grad[i] = (fPlus - fMinus) / (2 * eps)
	}
	return grad
}

func TestMaxPool2DForwardBackward(t *testing.T) {
	inputVals := []float64{
		1, 2, 3,
		4, 5, 6,
	}
	input := MustNew(inputVals, 1, 1, 2, 3)
	input.SetRequiresGrad(true)

	out, err := MaxPool2D(input, 2, 2, 1, 1, 0, 0)
	if err != nil {
		t.Fatalf("MaxPool2D returned error: %v", err)
	}

	expected := []float64{5, 6}
	if !almostEqualSlices(out.Data(), expected, 1e-9) {
		t.Fatalf("unexpected maxpool output: got %v want %v", out.Data(), expected)
	}

	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	grad := input.Grad()
	if grad == nil {
		t.Fatalf("expected gradient on input")
	}
	expectedGrad := []float64{
		0, 0, 0,
		0, 1, 1,
	}
	if !almostEqualSlices(grad.Data(), expectedGrad, 1e-9) {
		t.Fatalf("unexpected maxpool grad: got %v want %v", grad.Data(), expectedGrad)
	}
}

func TestAvgPool2DForwardBackward(t *testing.T) {
	inputVals := []float64{
		1, 2,
		3, 4,
		5, 6,
		7, 8,
	}
	input := MustNew(inputVals, 1, 1, 4, 2)
	input.SetRequiresGrad(true)

	out, err := AvgPool2D(input, 2, 2, 1, 1, 0, 0)
	if err != nil {
		t.Fatalf("AvgPool2D returned error: %v", err)
	}

	expected := []float64{2.5, 4.5, 6.5}
	if !almostEqualSlices(out.Data(), expected, 1e-9) {
		t.Fatalf("unexpected avgpool output: got %v want %v", out.Data(), expected)
	}

	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	grad := input.Grad()
	if grad == nil {
		t.Fatalf("expected gradient on input")
	}
	expectedGrad := []float64{
		0.25, 0.25,
		0.5, 0.5,
		0.5, 0.5,
		0.25, 0.25,
	}
	if !almostEqualSlices(grad.Data(), expectedGrad, 1e-9) {
		t.Fatalf("unexpected avgpool grad: got %v want %v", grad.Data(), expectedGrad)
	}
}

func layerNormLoss(inputVals, weightVals, biasVals []float64) float64 {
	inCopy := append([]float64(nil), inputVals...)
	wCopy := append([]float64(nil), weightVals...)
	bCopy := append([]float64(nil), biasVals...)
	input := MustNew(inCopy, 2, 3)
	weight := MustNew(wCopy, 3)
	bias := MustNew(bCopy, 3)
	out, err := LayerNorm(input, []int{3}, weight, bias, 1e-5)
	if err != nil {
		panic(err)
	}
	return Sum(out).Data()[0]
}

func TestLayerNormForwardBackward(t *testing.T) {
	inputVals := []float64{1, 2, 3, 4, 5, 6}
	weightVals := []float64{1.1, 0.9, -0.3}
	biasVals := []float64{0.2, -0.1, 0.05}

	input := MustNew(append([]float64(nil), inputVals...), 2, 3)
	input.SetRequiresGrad(true)
	weight := MustNew(append([]float64(nil), weightVals...), 3)
	weight.SetRequiresGrad(true)
	bias := MustNew(append([]float64(nil), biasVals...), 3)
	bias.SetRequiresGrad(true)

	out, err := LayerNorm(input, []int{3}, weight, bias, 1e-5)
	if err != nil {
		t.Fatalf("LayerNorm failed: %v", err)
	}

	// Forward check against manual computation
	expected := make([]float64, len(inputVals))
	normSize := 3
	for row := 0; row < 2; row++ {
		offset := row * normSize
		mean := 0.0
		for j := 0; j < normSize; j++ {
			mean += inputVals[offset+j]
		}
		mean /= float64(normSize)
		varSum := 0.0
		for j := 0; j < normSize; j++ {
			diff := inputVals[offset+j] - mean
			varSum += diff * diff
		}
		invStd := 1.0 / math.Sqrt(varSum/float64(normSize)+1e-5)
		for j := 0; j < normSize; j++ {
			xhat := (inputVals[offset+j] - mean) * invStd
			val := xhat*weightVals[j] + biasVals[j]
			expected[offset+j] = val
		}
	}
	if !almostEqualSlices(out.Data(), expected, 1e-7) {
		t.Fatalf("LayerNorm forward mismatch: got %v want %v", out.Data(), expected)
	}

	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	gInput := input.Grad().Data()
	numericInput := numericalGrad(func(vals []float64) float64 {
		return layerNormLoss(vals, weightVals, biasVals)
	}, inputVals, 1e-5)
	if !almostEqualSlices(gInput, numericInput, 1e-4) {
		t.Fatalf("LayerNorm input grad mismatch: got %v want %v", gInput, numericInput)
	}

	gWeight := weight.Grad().Data()
	numericWeight := numericalGrad(func(vals []float64) float64 {
		return layerNormLoss(inputVals, vals, biasVals)
	}, weightVals, 1e-5)
	if !almostEqualSlices(gWeight, numericWeight, 1e-4) {
		t.Fatalf("LayerNorm weight grad mismatch: got %v want %v", gWeight, numericWeight)
	}

	gBias := bias.Grad().Data()
	numericBias := numericalGrad(func(vals []float64) float64 {
		return layerNormLoss(inputVals, weightVals, vals)
	}, biasVals, 1e-5)
	if !almostEqualSlices(gBias, numericBias, 1e-4) {
		t.Fatalf("LayerNorm bias grad mismatch: got %v want %v", gBias, numericBias)
	}
}

func batchNormLoss(inputVals, weightVals, biasVals, runningMeanVals, runningVarVals []float64, momentum, eps float64) float64 {
	inCopy := append([]float64(nil), inputVals...)
	rmCopy := append([]float64(nil), runningMeanVals...)
	rvCopy := append([]float64(nil), runningVarVals...)
	input := MustNew(inCopy, 2, 2)
	runningMean := MustNew(rmCopy, 2)
	runningVar := MustNew(rvCopy, 2)
	var weight *Tensor
	if weightVals != nil {
		weight = MustNew(append([]float64(nil), weightVals...), 2)
	}
	var bias *Tensor
	if biasVals != nil {
		bias = MustNew(append([]float64(nil), biasVals...), 2)
	}
	out, err := BatchNorm(input, runningMean, runningVar, weight, bias, momentum, eps, true)
	if err != nil {
		panic(err)
	}
	return Sum(out).Data()[0]
}

func TestBatchNormTrainingAndEval(t *testing.T) {
	inputVals := []float64{
		1, 2,
		3, 6,
	}
	weightVals := []float64{0.5, -1.0}
	biasVals := []float64{0.1, 0.2}
	runningMeanVals := []float64{0, 0}
	runningVarVals := []float64{1, 1}
	momentum := 0.1
	eps := 1e-5

	input := MustNew(append([]float64(nil), inputVals...), 2, 2)
	input.SetRequiresGrad(true)
	runningMean := MustNew(append([]float64(nil), runningMeanVals...), 2)
	runningVar := MustNew(append([]float64(nil), runningVarVals...), 2)
	weight := MustNew(append([]float64(nil), weightVals...), 2)
	weight.SetRequiresGrad(true)
	bias := MustNew(append([]float64(nil), biasVals...), 2)
	bias.SetRequiresGrad(true)

	out, err := BatchNorm(input, runningMean, runningVar, weight, bias, momentum, eps, true)
	if err != nil {
		t.Fatalf("BatchNorm training failed: %v", err)
	}

	// Forward expected (only rank-2 case)
	count := float64(len(inputVals) / 2)
	savedMean := make([]float64, 2)
	savedVar := make([]float64, 2)
	expected := make([]float64, len(inputVals))
	for c := 0; c < 2; c++ {
		sum := 0.0
		for n := 0; n < 2; n++ {
			sum += inputVals[n*2+c]
		}
		mean := sum / count
		savedMean[c] = mean
		varSum := 0.0
		for n := 0; n < 2; n++ {
			diff := inputVals[n*2+c] - mean
			varSum += diff * diff
		}
		varVal := varSum / count
		savedVar[c] = varVal
		invStd := 1.0 / math.Sqrt(varVal+eps)
		for n := 0; n < 2; n++ {
			xhat := (inputVals[n*2+c] - mean) * invStd
			val := xhat
			val *= weightVals[c]
			val += biasVals[c]
			expected[n*2+c] = val
		}
	}
	if !almostEqualSlices(out.Data(), expected, 1e-6) {
		t.Fatalf("BatchNorm forward mismatch: got %v want %v", out.Data(), expected)
	}

	expectedRunningMean := []float64{
		(1-momentum)*runningMeanVals[0] + momentum*savedMean[0],
		(1-momentum)*runningMeanVals[1] + momentum*savedMean[1],
	}
	expectedRunningVar := []float64{
		(1-momentum)*runningVarVals[0] + momentum*savedVar[0],
		(1-momentum)*runningVarVals[1] + momentum*savedVar[1],
	}
	if !almostEqualSlices(runningMean.Data(), expectedRunningMean, 1e-6) {
		t.Fatalf("runningMean mismatch: got %v want %v", runningMean.Data(), expectedRunningMean)
	}
	if !almostEqualSlices(runningVar.Data(), expectedRunningVar, 1e-6) {
		t.Fatalf("runningVar mismatch: got %v want %v", runningVar.Data(), expectedRunningVar)
	}

	s := Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	numericInput := numericalGrad(func(vals []float64) float64 {
		return batchNormLoss(vals, weightVals, biasVals, runningMeanVals, runningVarVals, momentum, eps)
	}, inputVals, 1e-5)
	if !almostEqualSlices(input.Grad().Data(), numericInput, 1e-4) {
		t.Fatalf("BatchNorm input grad mismatch: got %v want %v", input.Grad().Data(), numericInput)
	}

	numericWeight := numericalGrad(func(vals []float64) float64 {
		return batchNormLoss(inputVals, vals, biasVals, runningMeanVals, runningVarVals, momentum, eps)
	}, weightVals, 1e-5)
	if !almostEqualSlices(weight.Grad().Data(), numericWeight, 1e-4) {
		t.Fatalf("BatchNorm weight grad mismatch: got %v want %v", weight.Grad().Data(), numericWeight)
	}

	numericBias := numericalGrad(func(vals []float64) float64 {
		return batchNormLoss(inputVals, weightVals, vals, runningMeanVals, runningVarVals, momentum, eps)
	}, biasVals, 1e-5)
	if !almostEqualSlices(bias.Grad().Data(), numericBias, 1e-4) {
		t.Fatalf("BatchNorm bias grad mismatch: got %v want %v", bias.Grad().Data(), numericBias)
	}

	// Eval path should reuse updated running stats
	evalInput := MustNew(append([]float64(nil), inputVals...), 2, 2)
	evalOut, err := BatchNorm(evalInput, runningMean.Clone(), runningVar.Clone(), weight.Detach(), bias.Detach(), momentum, eps, false)
	if err != nil {
		t.Fatalf("BatchNorm eval failed: %v", err)
	}
	expectedEval := make([]float64, len(inputVals))
	for c := 0; c < 2; c++ {
		invStd := 1.0 / math.Sqrt(runningVar.Data()[c]+eps)
		for n := 0; n < 2; n++ {
			xhat := (inputVals[n*2+c] - runningMean.Data()[c]) * invStd
			val := xhat*weightVals[c] + biasVals[c]
			expectedEval[n*2+c] = val
		}
	}
	if !almostEqualSlices(evalOut.Data(), expectedEval, 1e-6) {
		t.Fatalf("BatchNorm eval mismatch: got %v want %v", evalOut.Data(), expectedEval)
	}
}
