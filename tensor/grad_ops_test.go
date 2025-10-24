package tensor

import "testing"

func TestGradOps(t *testing.T) {
	a := MustNew([]float64{1, 2, 3, 4}, 2, 2)
	a.SetRequiresGrad(true)
	sum := Sum(a)
	if err := sum.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	if got := a.GradPowSum(2); got != 4 {
		t.Fatalf("GradPowSum mismatch: got %v want 4", got)
	}
	a.ScaleGrad(0.5)
	if grad := a.Grad(); grad == nil || !AlmostEqualSlices(grad.Data(), []float64{0.5, 0.5, 0.5, 0.5}, 1e-9) {
		t.Fatalf("unexpected grad after scaling: %v", grad)
	}

	a.ZeroGrad()
	if a.GradPowSum(2) != 0 {
		t.Fatalf("expected zero grad after ZeroGrad")
	}

	clipInput := MustNew([]float64{3, -4, 0.5}, 3)
	clipInput.SetRequiresGrad(true)
	sq, err := Mul(clipInput, clipInput)
	if err != nil {
		t.Fatalf("mul failed: %v", err)
	}
	clipLoss := Sum(sq)
	if err := clipLoss.Backward(); err != nil {
		t.Fatalf("clip backward failed: %v", err)
	}
	clipInput.ClipGradValue(2)
	if grad := clipInput.Grad(); grad == nil || !AlmostEqualSlices(grad.Data(), []float64{2, -2, 1}, 1e-9) {
		t.Fatalf("unexpected clipped grad: %v", grad)
	}
}
