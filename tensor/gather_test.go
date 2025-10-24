package tensor

import "testing"

func TestGatherForwardBackward(t *testing.T) {
	input := MustNew([]float64{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)
	index := MustNew([]float64{
		2, 1, 0,
		0, 2, 1,
	}, 2, 3)
	input.SetRequiresGrad(true)
	out, err := Gather(input, 1, index)
	if err != nil {
		t.Fatalf("gather failed: %v", err)
	}
	want := []float64{3, 2, 1, 4, 6, 5}
	if !AlmostEqualSlices(out.Data(), want, 1e-9) {
		t.Fatalf("unexpected gather output: got %v want %v", out.Data(), want)
	}
	sum := Sum(out)
	if err := sum.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	grad := input.Grad()
	if grad == nil {
		t.Fatalf("expected grad on input")
	}
	wantGrad := []float64{1, 1, 1, 1, 1, 1}
	if !AlmostEqualSlices(grad.Data(), wantGrad, 1e-9) {
		t.Fatalf("unexpected grad: got %v want %v", grad.Data(), wantGrad)
	}
}

func TestGatherIndexValidation(t *testing.T) {
	input := MustNew([]float64{1, 2, 3}, 3)
	mismatch := MustNew([]float64{0, 1}, 1, 2)
	if _, err := Gather(input, 0, mismatch); err == nil {
		t.Fatalf("expected rank mismatch error")
	}
	badIndex := MustNew([]float64{4, 0, 1}, 3)
	if _, err := Gather(input, 0, badIndex); err == nil {
		t.Fatalf("expected index out of range error")
	}
}
