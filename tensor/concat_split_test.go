package tensor

import "testing"

func TestConcatSplitRoundTrip(t *testing.T) {
	a := MustNew([]float64{1, 2, 3, 4}, 2, 2)
	b := MustNew([]float64{5, 6, 7, 8}, 2, 2)
	a.SetRequiresGrad(true)
	b.SetRequiresGrad(true)
	cat, err := Concat(0, a, b)
	if err != nil {
		t.Fatalf("concat failed: %v", err)
	}
	if !AlmostEqualSlices(cat.Data(), []float64{1, 2, 3, 4, 5, 6, 7, 8}, 1e-9) {
		t.Fatalf("unexpected concat data: %v", cat.Data())
	}
	parts, err := Split(0, []int{2, 2}, cat)
	if err != nil {
		t.Fatalf("split failed: %v", err)
	}
	if len(parts) != 2 {
		t.Fatalf("expected 2 parts, got %d", len(parts))
	}
	recA := parts[0]
	recB := parts[1]
	if !AlmostEqualSlices(recA.Data(), a.Data(), 1e-9) || !AlmostEqualSlices(recB.Data(), b.Data(), 1e-9) {
		t.Fatalf("split data mismatch")
	}

	sum := Sum(cat)
	if err := sum.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	if grad := a.Grad(); grad == nil || !AlmostEqualSlices(grad.Data(), []float64{1, 1, 1, 1}, 1e-9) {
		t.Fatalf("unexpected grad for a: %v", grad)
	}
	if grad := b.Grad(); grad == nil || !AlmostEqualSlices(grad.Data(), []float64{1, 1, 1, 1}, 1e-9) {
		t.Fatalf("unexpected grad for b: %v", grad)
	}
}

func TestConcatErrors(t *testing.T) {
	if _, err := Concat(0); err == nil {
		t.Fatalf("expected error for empty tensors")
	}
	a := MustNew([]float64{1, 2}, 2, 1)
	b := MustNew([]float64{3, 4}, 1, 2)
	if _, err := Concat(0, a, b); err == nil {
		t.Fatalf("expected shape mismatch error")
	}
}

func TestSplitErrors(t *testing.T) {
	t1 := MustNew([]float64{1, 2, 3, 4}, 2, 2)
	if _, err := Split(0, []int{}, t1); err == nil {
		t.Fatalf("expected error for empty sizes")
	}
	if _, err := Split(2, []int{1, 1}, t1); err == nil {
		t.Fatalf("expected axis out of range")
	}
	if _, err := Split(0, []int{1, 1, 1}, t1); err == nil {
		t.Fatalf("expected size mismatch error")
	}
}
