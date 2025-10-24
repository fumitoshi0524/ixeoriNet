package nn

import (
	"path/filepath"
	"testing"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func TestSaveAndLoadModule(t *testing.T) {
	lin1 := NewLinear(2, 2, true)
	lin2 := NewLinear(2, 1, true)
	if err := lin1.Weight().SetData([]float64{0.1, -0.2, 0.3, -0.4}); err != nil {
		t.Fatalf("set lin1 weight: %v", err)
	}
	if err := lin1.Bias().SetData([]float64{0.05, -0.05}); err != nil {
		t.Fatalf("set lin1 bias: %v", err)
	}
	if err := lin2.Weight().SetData([]float64{0.6, -0.8}); err != nil {
		t.Fatalf("set lin2 weight: %v", err)
	}
	if err := lin2.Bias().SetData([]float64{0.2}); err != nil {
		t.Fatalf("set lin2 bias: %v", err)
	}
	model := NewSequential(lin1, Relu(), lin2)

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.json")
	if err := SaveModule(path, model); err != nil {
		t.Fatalf("SaveModule failed: %v", err)
	}

	// Overwrite parameters to confirm load restores them.
	mustSetData(t, lin1.Weight(), []float64{1, 1, 1, 1})
	mustSetData(t, lin1.Bias(), []float64{1, 1})
	mustSetData(t, lin2.Weight(), []float64{-1, -1})
	mustSetData(t, lin2.Bias(), []float64{-1})

	if err := LoadModule(path, model); err != nil {
		t.Fatalf("LoadModule failed: %v", err)
	}
	wantLin1W := []float64{0.1, -0.2, 0.3, -0.4}
	wantLin1B := []float64{0.05, -0.05}
	wantLin2W := []float64{0.6, -0.8}
	wantLin2B := []float64{0.2}
	if !floatsAlmostEqual(lin1.Weight().Data(), wantLin1W, 1e-9) {
		t.Fatalf("lin1 weight mismatch after load")
	}
	if !floatsAlmostEqual(lin1.Bias().Data(), wantLin1B, 1e-9) {
		t.Fatalf("lin1 bias mismatch after load")
	}
	if !floatsAlmostEqual(lin2.Weight().Data(), wantLin2W, 1e-9) {
		t.Fatalf("lin2 weight mismatch after load")
	}
	if !floatsAlmostEqual(lin2.Bias().Data(), wantLin2B, 1e-9) {
		t.Fatalf("lin2 bias mismatch after load")
	}
}

func TestSaveModuleErrorsForStateless(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "stateless.json")
	if err := SaveModule(path, Relu()); err == nil {
		t.Fatalf("expected error when saving stateless module")
	}
}

func TestZeroGradAllHandlesNil(t *testing.T) {
	lin := NewLinear(2, 2, true)
	lin.Weight().SetRequiresGrad(true)
	lin.Bias().SetRequiresGrad(true)

	input := tensor.MustNew([]float64{1, -1, 2, -2}, 2, 2)
	out, err := lin.Forward(input)
	if err != nil {
		t.Fatalf("linear forward failed: %v", err)
	}
	loss := tensor.Sum(out)
	if err := loss.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	if lin.Weight().Grad() == nil {
		t.Fatalf("expected grad before ZeroGradAll")
	}

	ZeroGradAll(nil, lin)
	if lin.Weight().Grad() != nil || lin.Bias().Grad() != nil {
		t.Fatalf("ZeroGradAll should clear grads even with nil module present")
	}
}
