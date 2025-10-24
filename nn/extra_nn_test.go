package nn

import (
	"math"
	"testing"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func TestDropoutTrainVsEval(t *testing.T) {
	d := NewDropout(0.5)
	input := tensor.MustNew([]float64{1, 2, 3, 4}, 2, 2)
	input.SetRequiresGrad(true)
	d.Train()
	trainOut, err := d.Forward(input)
	if err != nil {
		t.Fatalf("dropout forward (train) failed: %v", err)
	}
	if trainOut.Shape()[0] != 2 || trainOut.Shape()[1] != 2 {
		t.Fatalf("unexpected shape: %v", trainOut.Shape())
	}
	scale := 1.0 / (1 - 0.5)
	inputData := input.Data()
	for i, v := range trainOut.Data() {
		if v != 0 && math.Abs(v-inputData[i]*scale) > 1e-9 {
			t.Fatalf("train dropout mismatch at %d: got %v want 0 or %v", i, v, inputData[i]*scale)
		}
	}
	s := tensor.Sum(trainOut)
	if err := s.Backward(); err != nil {
		t.Fatalf("dropout backward failed: %v", err)
	}
	grad := input.Grad()
	if grad == nil {
		t.Fatalf("expected gradient for input")
	}
	for i, g := range grad.Data() {
		if g != 0 && math.Abs(g-scale) > 1e-9 {
			t.Fatalf("unexpected grad at %d: got %v", i, g)
		}
	}
	d.Eval()
	evalOut, err := d.Forward(input)
	if err != nil {
		t.Fatalf("dropout forward (eval) failed: %v", err)
	}
	if !floatsAlmostEqual(evalOut.Data(), input.Data(), 1e-9) {
		t.Fatalf("dropout eval should match input: got %v want %v", evalOut.Data(), input.Data())
	}
}

func TestEmbeddingForwardBackward(t *testing.T) {
	emb := NewEmbedding(5, 3)
	indices := tensor.MustNew([]float64{0, 2, 4}, 3)
	indices.SetRequiresGrad(false)
	out, err := emb.Forward(indices)
	if err != nil {
		t.Fatalf("embedding forward failed: %v", err)
	}
	if out.Shape()[0] != 3 || out.Shape()[1] != 3 {
		t.Fatalf("unexpected embedding shape: %v", out.Shape())
	}
	s := tensor.Sum(out)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	grad := emb.weight.Grad()
	if grad == nil {
		t.Fatalf("expected embedding weight grad")
	}
	if grad.Numel() != emb.weight.Numel() {
		t.Fatalf("embedding grad size mismatch")
	}
}

func TestPoolingModules(t *testing.T) {
	input := tensor.MustNew([]float64{1, 2, 3, 4}, 1, 1, 2, 2)
	mp := NewMaxPool2d(2, 2, 1, 1, 0, 0)
	maxOut, err := mp.Forward(input)
	if err != nil {
		t.Fatalf("maxpool forward failed: %v", err)
	}
	if maxOut.Numel() != 1 || math.Abs(maxOut.Data()[0]-4) > 1e-9 {
		t.Fatalf("unexpected maxpool result: %v", maxOut.Data())
	}

	ap := NewAvgPool2d(2, 2, 1, 1, 0, 0)
	avgOut, err := ap.Forward(input)
	if err != nil {
		t.Fatalf("avgpool forward failed: %v", err)
	}
	if avgOut.Numel() != 1 || math.Abs(avgOut.Data()[0]-2.5) > 1e-9 {
		t.Fatalf("unexpected avgpool result: %v", avgOut.Data())
	}
}
