package tensor

import (
	"math"
	"testing"
)

func TestLogSoftmaxForwardBackward(t *testing.T) {
	input := MustNew([]float64{1, 0, -1}, 1, 3)
	input.SetRequiresGrad(true)
	logsm, err := LogSoftmax(input, 1)
	if err != nil {
		t.Fatalf("logsoftmax forward failed: %v", err)
	}
	// Manual computation for reference
	vals := []float64{1, 0, -1}
	maxVal := 1.0
	sumExp := math.Exp(vals[0]-maxVal) + math.Exp(vals[1]-maxVal) + math.Exp(vals[2]-maxVal)
	expected := []float64{
		vals[0] - (maxVal + math.Log(sumExp)),
		vals[1] - (maxVal + math.Log(sumExp)),
		vals[2] - (maxVal + math.Log(sumExp)),
	}
	if !AlmostEqualSlices(logsm.Data(), expected, 1e-9) {
		t.Fatalf("logsoftmax output mismatch: %v", logsm.Data())
	}

	target := MustNew([]float64{0, 1, 0}, 1, 3)
	prod, err := Mul(logsm, target)
	if err != nil {
		t.Fatalf("mul failed: %v", err)
	}
	loss := MulScalar(Sum(prod), -1)
	if err := loss.Backward(); err != nil {
		t.Fatalf("logsoftmax backward failed: %v", err)
	}
	grad := input.Grad().Data()
	soft := make([]float64, 3)
	logsmVals := logsm.Data()
	for i := range soft {
		soft[i] = math.Exp(logsmVals[i])
	}
	expectedGrad := []float64{soft[0], soft[1] - 1, soft[2]}
	if !AlmostEqualSlices(grad, expectedGrad, 1e-9) {
		t.Fatalf("logsoftmax grad mismatch: %v", grad)
	}

	softmax, err := Softmax(input.Detach(), 1)
	if err != nil {
		t.Fatalf("softmax failed: %v", err)
	}
	expLog := Exp(logsm)
	if !AlmostEqualSlices(softmax.Data(), expLog.Data(), 1e-9) {
		t.Fatalf("softmax not exp(logsoftmax): %v vs %v", softmax.Data(), expLog.Data())
	}
}

func TestLogSoftmaxValidatesAxis(t *testing.T) {
	badInput := MustNew([]float64{1, 2, 3}, 3)
	if _, err := LogSoftmax(badInput, 0); err == nil {
		t.Fatalf("expected error for rank mismatch")
	}
	mat := MustNew([]float64{1, 2, 3, 4}, 2, 2)
	if _, err := LogSoftmax(mat, 0); err == nil {
		t.Fatalf("expected error for unsupported axis")
	}
}
