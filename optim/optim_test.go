package optim

import (
	"math"
	"testing"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func almostEqual(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := math.Abs(a[i] - b[i])
		if diff > tol {
			return false
		}
	}
	return true
}

func TestSGDStepAndMomentum(t *testing.T) {
	param := tensor.MustNew([]float64{1, -2}, 2)
	param.SetRequiresGrad(true)

	s := tensor.Sum(param)
	if err := s.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	opt := NewSGD([]*tensor.Tensor{param}, 0.1, 0)
	if err := opt.Step(); err != nil {
		t.Fatalf("sgd step failed: %v", err)
	}
	expected := []float64{0.9, -2.1}
	if !almostEqual(param.Data(), expected, 1e-9) {
		t.Fatalf("unexpected param after SGD step: got %v want %v", param.Data(), expected)
	}

	// Momentum run for two additional updates with constant gradients of ones.
	paramZero := tensor.MustNew([]float64{1, -2}, 2)
	paramZero.SetRequiresGrad(true)
	momentumOpt := NewSGD([]*tensor.Tensor{paramZero}, 0.1, 0.5)
	for i := 0; i < 2; i++ {
		momentumOpt.ZeroGrad()
		s := tensor.Sum(paramZero)
		if err := s.Backward(); err != nil {
			t.Fatalf("momentum backward failed: %v", err)
		}
		if err := momentumOpt.Step(); err != nil {
			t.Fatalf("momentum step failed: %v", err)
		}
	}
	expectedMomentum := []float64{0.75, -2.25}
	if !almostEqual(paramZero.Data(), expectedMomentum, 1e-9) {
		t.Fatalf("unexpected param after momentum SGD: got %v want %v", paramZero.Data(), expectedMomentum)
	}
}

func TestAdamConvergesOnQuadratic(t *testing.T) {
	param := tensor.MustNew([]float64{5}, 1)
	param.SetRequiresGrad(true)
	target := tensor.Full(3, 1)
	opt := NewAdam([]*tensor.Tensor{param}, 0.05, 0.9, 0.999, 1e-8)

	for i := 0; i < 200; i++ {
		opt.ZeroGrad()
		diff, err := tensor.Sub(param, target)
		if err != nil {
			t.Fatalf("sub failed: %v", err)
		}
		sq := tensor.Pow(diff, 2)
		loss := tensor.Mean(sq)
		if err := loss.Backward(); err != nil {
			t.Fatalf("adam backward failed: %v", err)
		}
		if err := opt.Step(); err != nil {
			t.Fatalf("adam step failed: %v", err)
		}
	}

	val := param.Data()[0]
	if math.Abs(val-3) > 1e-2 {
		t.Fatalf("adam did not converge close to target: got %.6f", val)
	}
}

func TestAdagradConverges(t *testing.T) {
	param := tensor.MustNew([]float64{2}, 1)
	param.SetRequiresGrad(true)
	target := tensor.Full(0, 1)
	opt := NewAdagrad([]*tensor.Tensor{param}, 0.5, 1e-8)

	for i := 0; i < 200; i++ {
		opt.ZeroGrad()
		diff, err := tensor.Sub(param, target)
		if err != nil {
			t.Fatalf("sub failed: %v", err)
		}
		sq := tensor.Pow(diff, 2)
		loss := tensor.Mean(sq)
		if err := loss.Backward(); err != nil {
			t.Fatalf("adagrad backward failed: %v", err)
		}
		if err := opt.Step(); err != nil {
			t.Fatalf("adagrad step failed: %v", err)
		}
	}

	if math.Abs(param.Data()[0]) > 1e-2 {
		t.Fatalf("adagrad did not converge: got %.6f", param.Data()[0])
	}
}

func TestAdadeltaConverges(t *testing.T) {
	param := tensor.MustNew([]float64{3}, 1)
	param.SetRequiresGrad(true)
	target := tensor.Full(-1, 1)
	opt := NewAdadelta([]*tensor.Tensor{param}, 1.0, 0.9, 1e-6)

	var initialLoss float64
	var finalLoss float64
	for i := 0; i < 300; i++ {
		opt.ZeroGrad()
		diff, err := tensor.Sub(param, target)
		if err != nil {
			t.Fatalf("sub failed: %v", err)
		}
		sq := tensor.Pow(diff, 2)
		loss := tensor.Mean(sq)
		lossVal := loss.Data()[0]
		if i == 0 {
			initialLoss = lossVal
		}
		finalLoss = lossVal
		if err := loss.Backward(); err != nil {
			t.Fatalf("adadelta backward failed: %v", err)
		}
		if err := opt.Step(); err != nil {
			t.Fatalf("adadelta step failed: %v", err)
		}
	}

	if !(finalLoss < initialLoss) {
		t.Fatalf("adadelta loss did not decrease: initial=%.6f final=%.6f", initialLoss, finalLoss)
	}
}

func TestAdamWConverges(t *testing.T) {
	param := tensor.MustNew([]float64{4}, 1)
	param.SetRequiresGrad(true)
	target := tensor.Full(1, 1)
	cfg := AdamWConfig{LR: 0.05, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8, WeightDecay: 0}
	opt := NewAdamWWithConfig([]*tensor.Tensor{param}, cfg)

	for i := 0; i < 200; i++ {
		opt.ZeroGrad()
		diff, err := tensor.Sub(param, target)
		if err != nil {
			t.Fatalf("sub failed: %v", err)
		}
		sq := tensor.Pow(diff, 2)
		loss := tensor.Mean(sq)
		if err := loss.Backward(); err != nil {
			t.Fatalf("adamw backward failed: %v", err)
		}
		if err := opt.Step(); err != nil {
			t.Fatalf("adamw step failed: %v", err)
		}
	}

	if math.Abs(param.Data()[0]-1) > 1e-2 {
		t.Fatalf("adamw did not converge: got %.6f", param.Data()[0])
	}
}

func TestRMSPropConverges(t *testing.T) {
	param := tensor.MustNew([]float64{-3}, 1)
	param.SetRequiresGrad(true)
	target := tensor.Full(2, 1)
	cfg := RMSPropConfig{LR: 0.05, Alpha: 0.99, Eps: 1e-8, WeightDecay: 0, Momentum: 0}
	opt := NewRMSPropWithConfig([]*tensor.Tensor{param}, cfg)

	for i := 0; i < 400; i++ {
		opt.ZeroGrad()
		diff, err := tensor.Sub(param, target)
		if err != nil {
			t.Fatalf("sub failed: %v", err)
		}
		sq := tensor.Pow(diff, 2)
		loss := tensor.Mean(sq)
		if err := loss.Backward(); err != nil {
			t.Fatalf("rmsprop backward failed: %v", err)
		}
		if err := opt.Step(); err != nil {
			t.Fatalf("rmsprop step failed: %v", err)
		}
	}

	if math.Abs(param.Data()[0]-2) > 5e-2 {
		t.Fatalf("rmsprop did not approach target: got %.6f", param.Data()[0])
	}
}

func TestGradientClippingUtilities(t *testing.T) {
	param := tensor.MustNew([]float64{3, 4}, 2)
	param.SetRequiresGrad(true)
	sum := tensor.Sum(param)
	if err := sum.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}

	originalNorm := ClipGradNorm([]*tensor.Tensor{param}, 1.0, 2)
	if math.Abs(originalNorm-math.Sqrt(2)) > 1e-6 {
		t.Fatalf("unexpected original norm: %.6f", originalNorm)
	}
	grad := param.Grad()
	data := grad.Data()
	clippedNorm := math.Sqrt(data[0]*data[0] + data[1]*data[1])
	if math.Abs(clippedNorm-1) > 1e-6 {
		t.Fatalf("ClipGradNorm did not rescale to 1, got %.6f", clippedNorm)
	}

	param.ZeroGrad()
	sum = tensor.Sum(param)
	if err := sum.Backward(); err != nil {
		t.Fatalf("backward failed: %v", err)
	}
	ClipGradValue([]*tensor.Tensor{param}, 0.5)
	grad = param.Grad()
	for i, v := range grad.Data() {
		if math.Abs(v) > 0.5+1e-9 {
			t.Fatalf("ClipGradValue exceeded limit at %d: %.6f", i, v)
		}
	}
}

func TestMaxNormConstraint(t *testing.T) {
	param := tensor.MustNew([]float64{3, 4}, 2)
	constraint := NewMaxNormConstraint(1.0, 2)
	if err := constraint.Apply(param); err != nil {
		t.Fatalf("constraint apply failed: %v", err)
	}
	data := param.Data()
	norm := math.Sqrt(data[0]*data[0] + data[1]*data[1])
	if norm > 1.0+1e-6 {
		t.Fatalf("max norm constraint violated: %.6f", norm)
	}
}
