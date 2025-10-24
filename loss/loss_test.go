package loss

import (
	"math"
	"testing"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func TestMSEForwardBackward(t *testing.T) {
	pred := tensor.MustNew([]float64{1, 3}, 2, 1)
	pred.SetRequiresGrad(true)
	target := tensor.MustNew([]float64{2, 1}, 2, 1)

	l, err := MSE(pred, target)
	if err != nil {
		t.Fatalf("MSE returned error: %v", err)
	}
	expectedLoss := (math.Pow(1-2, 2) + math.Pow(3-1, 2)) / 2
	if math.Abs(l.Data()[0]-expectedLoss) > 1e-9 {
		t.Fatalf("unexpected MSE value: got %v want %v", l.Data()[0], expectedLoss)
	}

	if err := l.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	grad := pred.Grad()
	if grad == nil {
		t.Fatalf("expected gradient on predictions")
	}
	expectedGrad := []float64{-1, 2}
	if len(grad.Data()) != len(expectedGrad) {
		t.Fatalf("gradient size mismatch: got %d want %d", len(grad.Data()), len(expectedGrad))
	}
	for i, v := range grad.Data() {
		if math.Abs(v-expectedGrad[i]) > 1e-9 {
			t.Fatalf("unexpected grad at %d: got %v want %v", i, v, expectedGrad[i])
		}
	}
}
