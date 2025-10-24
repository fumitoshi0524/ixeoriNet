package nn

import (
	"testing"

	"github.com/fumitoshi0524/ixeoriNet/loss"
	"github.com/fumitoshi0524/ixeoriNet/optim"
	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func floatsAlmostEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}

func TestSequentialForwardBackward(t *testing.T) {
	linear1 := NewLinear(3, 2, true)
	if err := linear1.Weight().SetData([]float64{
		0.5, -1.0, 1.5,
		-0.25, 0.75, -0.5,
	}); err != nil {
		t.Fatalf("set linear1 weight: %v", err)
	}
	if err := linear1.Bias().SetData([]float64{0.1, -0.2}); err != nil {
		t.Fatalf("set linear1 bias: %v", err)
	}
	relu := Relu()
	linear2 := NewLinear(2, 1, true)
	if err := linear2.Weight().SetData([]float64{0.6, -1.2}); err != nil {
		t.Fatalf("set linear2 weight: %v", err)
	}
	if err := linear2.Bias().SetData([]float64{0.05}); err != nil {
		t.Fatalf("set linear2 bias: %v", err)
	}
	model := NewSequential(linear1, relu, linear2)

	inputs := tensor.MustNew([]float64{
		1, 0, -1,
		2, 1, 0,
	}, 2, 3)
	targets := tensor.MustNew([]float64{1, -1}, 2, 1)

	out, err := model.Forward(inputs)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}
	if out == nil {
		t.Fatalf("forward returned nil output")
	}

	l, err := loss.MSE(out, targets)
	if err != nil {
		t.Fatalf("loss failed: %v", err)
	}
	if err := l.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	if linear1.Weight().Grad() == nil {
		t.Fatalf("expected gradient on linear1 weight")
	}
	if linear1.Bias().Grad() == nil {
		t.Fatalf("expected gradient on linear1 bias")
	}
	if linear2.Weight().Grad() == nil {
		t.Fatalf("expected gradient on linear2 weight")
	}
	if linear2.Bias().Grad() == nil {
		t.Fatalf("expected gradient on linear2 bias")
	}

	ZeroGradAll(model)
	if linear1.Weight().Grad() != nil || linear2.Weight().Grad() != nil {
		t.Fatalf("ZeroGradAll did not clear gradients")
	}
}

func TestSequentialStateDictAndLoad(t *testing.T) {
	linear1 := NewLinear(3, 2, true)
	linear2 := NewLinear(2, 1, true)
	model := NewSequential(linear1, Relu(), linear2)

	if err := linear1.Weight().SetData([]float64{
		0.5, -0.25, 0.75,
		-1.2, 0.9, -0.4,
	}); err != nil {
		t.Fatalf("set linear1 weight: %v", err)
	}
	if err := linear1.Bias().SetData([]float64{0.3, -0.1}); err != nil {
		t.Fatalf("set linear1 bias: %v", err)
	}
	if err := linear2.Weight().SetData([]float64{1.1, -0.7}); err != nil {
		t.Fatalf("set linear2 weight: %v", err)
	}
	if err := linear2.Bias().SetData([]float64{0.2}); err != nil {
		t.Fatalf("set linear2 bias: %v", err)
	}

	state := map[string]*tensor.Tensor{}
	model.StateDict("", state)
	if len(state) != 4 {
		t.Fatalf("expected 4 tensors in state dict, got %d", len(state))
	}

	clone := NewSequential(NewLinear(3, 2, true), Relu(), NewLinear(2, 1, true))
	if err := clone.LoadState("", state); err != nil {
		t.Fatalf("LoadState failed: %v", err)
	}

	origParams := model.Parameters()
	cloneParams := clone.Parameters()
	if len(origParams) != len(cloneParams) {
		t.Fatalf("parameter length mismatch: %d vs %d", len(origParams), len(cloneParams))
	}
	for i := range origParams {
		if !floatsAlmostEqual(origParams[i].Data(), cloneParams[i].Data(), 1e-9) {
			t.Fatalf("parameter %d mismatch after load", i)
		}
	}
}

func TestSequentialTrainingWithSGD(t *testing.T) {
	linear := NewLinear(1, 1, true)
	if err := linear.Weight().SetData([]float64{0}); err != nil {
		t.Fatalf("set linear weight: %v", err)
	}
	if err := linear.Bias().SetData([]float64{0}); err != nil {
		t.Fatalf("set linear bias: %v", err)
	}
	model := NewSequential(linear)
	inputs := tensor.MustNew([]float64{-2, -1, 0, 1, 2, 3}, 6, 1)
	targets := tensor.MustNew([]float64{-5, -3, -1, 1, 3, 5}, 6, 1)
	opt := optim.NewSGD(model.Parameters(), 0.1, 0)

	var initialLoss float64
	for epoch := 0; epoch < 50; epoch++ {
		opt.ZeroGrad()
		pred, err := model.Forward(inputs)
		if err != nil {
			t.Fatalf("forward failed: %v", err)
		}
		l, err := loss.MSE(pred, targets)
		if err != nil {
			t.Fatalf("loss failed: %v", err)
		}
		if epoch == 0 {
			initialLoss = l.Data()[0]
		}
		if err := l.Backward(); err != nil {
			t.Fatalf("backward failed: %v", err)
		}
		if err := opt.Step(); err != nil {
			t.Fatalf("optimizer step failed: %v", err)
		}
	}
	pred, err := model.Forward(inputs)
	if err != nil {
		t.Fatalf("forward failed: %v", err)
	}
	finalLoss, err := loss.MSE(pred, targets)
	if err != nil {
		t.Fatalf("loss failed: %v", err)
	}
	if finalLoss.Data()[0] >= initialLoss {
		t.Fatalf("expected loss to decrease: initial=%.6f final=%.6f", initialLoss, finalLoss.Data()[0])
	}
}
