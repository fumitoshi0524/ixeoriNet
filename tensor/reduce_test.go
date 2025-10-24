package tensor

import "testing"

func TestMaxMinReduce(t *testing.T) {
	input := MustNew([]float64{
		1, 4, 2,
		3, 0, -1,
	}, 2, 3)
	input.SetRequiresGrad(true)

	mx, err := Max(input, 1)
	if err != nil {
		t.Fatalf("max reduce failed: %v", err)
	}
	mn, err := Min(input, 1)
	if err != nil {
		t.Fatalf("min reduce failed: %v", err)
	}
	if !AlmostEqualSlices(mx.Data(), []float64{4, 3}, 1e-9) {
		t.Fatalf("unexpected max result: %v", mx.Data())
	}
	if !AlmostEqualSlices(mn.Data(), []float64{1, -1}, 1e-9) {
		t.Fatalf("unexpected min result: %v", mn.Data())
	}

	total := Sum(mx)
	if err := total.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	grad := input.Grad()
	if grad == nil {
		t.Fatalf("expected gradient on input")
	}
	expected := []float64{0, 1, 0, 1, 0, 0}
	if !AlmostEqualSlices(grad.Data(), expected, 1e-9) {
		t.Fatalf("unexpected max gradient: %v", grad.Data())
	}

	input.ZeroGrad()
	totalMin := Sum(mn)
	if err := totalMin.Backward(); err != nil {
		t.Fatalf("min backward failed: %v", err)
	}
	grad = input.Grad()
	expectedMin := []float64{1, 0, 0, 0, 0, 1}
	if !AlmostEqualSlices(grad.Data(), expectedMin, 1e-9) {
		t.Fatalf("unexpected min gradient: %v", grad.Data())
	}
}

func TestReduceErrors(t *testing.T) {
	tensor := MustNew([]float64{1, 2, 3, 4}, 2, 2)
	if _, err := Max(tensor, 2); err == nil {
		t.Fatalf("expected axis out of range")
	}
	if _, err := Min(tensor, -3); err == nil {
		t.Fatalf("expected negative axis out of range")
	}
}
