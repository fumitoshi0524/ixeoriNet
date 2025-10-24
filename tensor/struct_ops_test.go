package tensor

import "testing"

func TestStackForwardBackward(t *testing.T) {
	a := MustNew([]float64{1, 2, 3}, 3)
	b := MustNew([]float64{4, 5, 6}, 3)
	a.SetRequiresGrad(true)
	b.SetRequiresGrad(true)

	stacked, err := Stack(0, a, b)
	if err != nil {
		t.Fatalf("stack failed: %v", err)
	}
	if !AlmostEqualSlices(stacked.Data(), []float64{1, 2, 3, 4, 5, 6}, 1e-9) {
		t.Fatalf("stack data mismatch: %v", stacked.Data())
	}
	if err := Sum(stacked).Backward(); err != nil {
		t.Fatalf("stack backward failed: %v", err)
	}
	if !AlmostEqualSlices(a.Grad().Data(), []float64{1, 1, 1}, 1e-9) {
		t.Fatalf("stack grad mismatch for a: %v", a.Grad().Data())
	}
	if !AlmostEqualSlices(b.Grad().Data(), []float64{1, 1, 1}, 1e-9) {
		t.Fatalf("stack grad mismatch for b: %v", b.Grad().Data())
	}
	if _, err := Stack(0); err == nil {
		t.Fatalf("expected error for empty stack call")
	}
}

func TestSqueezeUnsqueezeBackward(t *testing.T) {
	t1 := MustNew([]float64{1, 2, 3, 4}, 1, 2, 1, 2)
	t1.SetRequiresGrad(true)
	squeezed, err := Squeeze(t1, 0, 2)
	if err != nil {
		t.Fatalf("squeeze failed: %v", err)
	}
	if !equalShapes(squeezed.Shape(), []int{2, 2}) {
		t.Fatalf("unexpected squeezed shape: %v", squeezed.Shape())
	}
	if err := Sum(squeezed).Backward(); err != nil {
		t.Fatalf("squeeze backward failed: %v", err)
	}
	if !AlmostEqualSlices(t1.Grad().Data(), []float64{1, 1, 1, 1}, 1e-9) {
		t.Fatalf("squeeze grad mismatch: %v", t1.Grad().Data())
	}

	t2 := MustNew([]float64{1, 2, 3, 4}, 2, 2)
	t2.SetRequiresGrad(true)
	unsqueezed, err := Unsqueeze(t2, 1)
	if err != nil {
		t.Fatalf("unsqueeze failed: %v", err)
	}
	if !equalShapes(unsqueezed.Shape(), []int{2, 1, 2}) {
		t.Fatalf("unexpected unsqueezed shape: %v", unsqueezed.Shape())
	}
	if err := Sum(unsqueezed).Backward(); err != nil {
		t.Fatalf("unsqueeze backward failed: %v", err)
	}
	if !AlmostEqualSlices(t2.Grad().Data(), []float64{1, 1, 1, 1}, 1e-9) {
		t.Fatalf("unsqueeze grad mismatch: %v", t2.Grad().Data())
	}

	if _, err := Squeeze(t2, 0); err == nil {
		t.Fatalf("expected squeeze axis error")
	}
	if _, err := Unsqueeze(t2, 5); err == nil {
		t.Fatalf("expected unsqueeze axis error")
	}
}

func TestTransposeForwardBackward(t *testing.T) {
	input := MustNew([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	input.SetRequiresGrad(true)
	tr, err := Transpose(input)
	if err != nil {
		t.Fatalf("transpose failed: %v", err)
	}
	if !AlmostEqualSlices(tr.Data(), []float64{1, 4, 2, 5, 3, 6}, 1e-9) {
		t.Fatalf("transpose data mismatch: %v", tr.Data())
	}
	if err := Sum(tr).Backward(); err != nil {
		t.Fatalf("transpose backward failed: %v", err)
	}
	if !AlmostEqualSlices(input.Grad().Data(), []float64{1, 1, 1, 1, 1, 1}, 1e-9) {
		t.Fatalf("transpose grad mismatch: %v", input.Grad().Data())
	}

	bad := MustNew([]float64{1, 2, 3}, 3)
	if _, err := Transpose(bad); err == nil {
		t.Fatalf("expected transpose rank error")
	}
}

func TestChunkSplitBehavior(t *testing.T) {
	base := MustNew([]float64{1, 2, 3, 4, 5, 6}, 3, 2)
	base.SetRequiresGrad(true)
	chunks, err := Chunk(0, 2, base)
	if err != nil {
		t.Fatalf("chunk failed: %v", err)
	}
	if len(chunks) != 2 {
		t.Fatalf("unexpected chunk count: %d", len(chunks))
	}
	if !equalShapes(chunks[0].Shape(), []int{2, 2}) || !equalShapes(chunks[1].Shape(), []int{1, 2}) {
		t.Fatalf("unexpected chunk shapes: %v %v", chunks[0].Shape(), chunks[1].Shape())
	}
	sum0 := Sum(chunks[0])
	sum1 := Sum(chunks[1])
	total, err := Add(sum0, sum1)
	if err != nil {
		t.Fatalf("add failed: %v", err)
	}
	if err := total.Backward(); err != nil {
		t.Fatalf("chunk backward failed: %v", err)
	}
	if !AlmostEqualSlices(base.Grad().Data(), []float64{1, 1, 1, 1, 1, 1}, 1e-9) {
		t.Fatalf("chunk grad mismatch: %v", base.Grad().Data())
	}

	if _, err := Chunk(3, 2, base); err == nil {
		t.Fatalf("expected chunk axis error")
	}
}
