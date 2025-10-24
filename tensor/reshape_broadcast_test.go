package tensor

import "testing"

func TestReshapeBackward(t *testing.T) {
	input := MustNew([]float64{1, 2, 3, 4, 5, 6}, 6)
	input.SetRequiresGrad(true)
	reshaped, err := input.Reshape(2, 3)
	if err != nil {
		t.Fatalf("reshape failed: %v", err)
	}
	if !equalShapes(reshaped.Shape(), []int{2, 3}) {
		t.Fatalf("unexpected reshape shape: %v", reshaped.Shape())
	}
	inferred, err := reshaped.Reshape(3, -1)
	if err != nil {
		t.Fatalf("reshape with -1 failed: %v", err)
	}
	if !equalShapes(inferred.Shape(), []int{3, 2}) {
		t.Fatalf("unexpected inferred shape: %v", inferred.Shape())
	}

	sum := Sum(inferred)
	if err := sum.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	grad := input.Grad()
	if grad == nil || !AlmostEqualSlices(grad.Data(), []float64{1, 1, 1, 1, 1, 1}, 1e-9) {
		t.Fatalf("unexpected grad after reshape: %v", grad)
	}
}

func TestFlattenPreservesData(t *testing.T) {
	input := MustNew([]float64{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	input.SetRequiresGrad(true)
	flat, err := Flatten(input)
	if err != nil {
		t.Fatalf("flatten failed: %v", err)
	}
	if !equalShapes(flat.Shape(), []int{1, 6}) {
		t.Fatalf("unexpected flat shape: %v", flat.Shape())
	}
	if !AlmostEqualSlices(flat.Data(), []float64{1, 2, 3, 4, 5, 6}, 1e-9) {
		t.Fatalf("flatten data mismatch: %v", flat.Data())
	}
	sum := Sum(flat)
	if err := sum.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	grad := input.Grad()
	if grad == nil || !AlmostEqualSlices(grad.Data(), []float64{1, 1, 1, 1, 1, 1}, 1e-9) {
		t.Fatalf("unexpected grad after flatten: %v", grad)
	}
}

func TestReduceToShape(t *testing.T) {
	grad := MustNew([]float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}, 3, 2, 2)
	reduced, err := ReduceToShape(grad, []int{2, 2})
	if err != nil {
		t.Fatalf("ReduceToShape failed: %v", err)
	}
	data := grad.Data()
	want := make([]float64, 4)
	idx := 0
	for row := 0; row < 2; row++ {
		for col := 0; col < 2; col++ {
			sumVal := 0.0
			for depth := 0; depth < 3; depth++ {
				offset := depth*4 + row*2 + col
				sumVal += data[offset]
			}
			want[idx] = sumVal
			idx++
		}
	}
	if !AlmostEqualSlices(reduced.Data(), want, 1e-9) {
		t.Fatalf("unexpected reduced data: %v", reduced.Data())
	}

	if _, err := ReduceToShape(grad, []int{2, 2, 2, 2}); err == nil {
		t.Fatalf("expected error for higher rank target")
	}
	if _, err := ReduceToShape(grad, []int{3, 3}); err == nil {
		t.Fatalf("expected error for incompatible shape")
	}
}
