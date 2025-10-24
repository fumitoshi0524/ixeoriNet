package tensor

import (
	"math"
	"testing"
)

func TestReluForwardBackward(t *testing.T) {
	input := MustNew([]float64{-1, 0, 2}, 3)
	input.SetRequiresGrad(true)
	out := Relu(input)
	if !AlmostEqualSlices(out.Data(), []float64{0, 0, 2}, 1e-9) {
		t.Fatalf("relu output mismatch: %v", out.Data())
	}
	loss := Sum(out)
	if err := loss.Backward(); err != nil {
		t.Fatalf("relu backward failed: %v", err)
	}
	grad := input.Grad().Data()
	if !AlmostEqualSlices(grad, []float64{0, 0, 1}, 1e-9) {
		t.Fatalf("relu grad mismatch: %v", grad)
	}
}

func TestSigmoidForwardBackward(t *testing.T) {
	input := MustNew([]float64{-2, 0, 2}, 3)
	input.SetRequiresGrad(true)
	out := Sigmoid(input)
	expected := make([]float64, 3)
	deriv := make([]float64, 3)
	for i, v := range []float64{-2, 0, 2} {
		expected[i] = 1 / (1 + math.Exp(-v))
		deriv[i] = expected[i] * (1 - expected[i])
	}
	if !AlmostEqualSlices(out.Data(), expected, 1e-9) {
		t.Fatalf("sigmoid output mismatch: %v", out.Data())
	}
	if err := Sum(out).Backward(); err != nil {
		t.Fatalf("sigmoid backward failed: %v", err)
	}
	if !AlmostEqualSlices(input.Grad().Data(), deriv, 1e-9) {
		t.Fatalf("sigmoid grad mismatch: %v", input.Grad().Data())
	}
}

func TestTanhForwardBackward(t *testing.T) {
	values := []float64{-1, 0, 1}
	input := MustNew(values, 3)
	input.SetRequiresGrad(true)
	out := Tanh(input)
	expected := make([]float64, len(values))
	deriv := make([]float64, len(values))
	for i, v := range values {
		expected[i] = math.Tanh(v)
		deriv[i] = 1 - expected[i]*expected[i]
	}
	if !AlmostEqualSlices(out.Data(), expected, 1e-9) {
		t.Fatalf("tanh output mismatch: %v", out.Data())
	}
	if err := Sum(out).Backward(); err != nil {
		t.Fatalf("tanh backward failed: %v", err)
	}
	if !AlmostEqualSlices(input.Grad().Data(), deriv, 1e-9) {
		t.Fatalf("tanh grad mismatch: %v", input.Grad().Data())
	}
}

func TestLeakyReluForwardBackward(t *testing.T) {
	alpha := 0.1
	input := MustNew([]float64{-2, 0, 3}, 3)
	input.SetRequiresGrad(true)
	out := LeakyRelu(input, alpha)
	expected := []float64{-0.2, 0, 3}
	if !AlmostEqualSlices(out.Data(), expected, 1e-9) {
		t.Fatalf("leaky relu output mismatch: %v", out.Data())
	}
	if err := Sum(out).Backward(); err != nil {
		t.Fatalf("leaky relu backward failed: %v", err)
	}
	grad := input.Grad().Data()
	expectedGrad := []float64{alpha, alpha, 1}
	if !AlmostEqualSlices(grad, expectedGrad, 1e-9) {
		t.Fatalf("leaky relu grad mismatch: %v", grad)
	}
}

func TestELUForwardBackward(t *testing.T) {
	alpha := 1.0
	input := MustNew([]float64{-1, 0, 1}, 3)
	input.SetRequiresGrad(true)
	out := ELU(input, alpha)
	expected := []float64{alpha * (math.Exp(-1) - 1), 0, 1}
	if !AlmostEqualSlices(out.Data(), expected, 1e-9) {
		t.Fatalf("elu output mismatch: %v", out.Data())
	}
	if err := Sum(out).Backward(); err != nil {
		t.Fatalf("elu backward failed: %v", err)
	}
	grad := input.Grad().Data()
	expectedGrad := []float64{expected[0] + alpha, alpha, 1}
	if !AlmostEqualSlices(grad, expectedGrad, 1e-9) {
		t.Fatalf("elu grad mismatch: %v", grad)
	}
}

func TestSoftplusForwardBackward(t *testing.T) {
	beta := 1.0
	input := MustNew([]float64{-2, 0, 2}, 3)
	input.SetRequiresGrad(true)
	out := Softplus(input, beta)
	expected := make([]float64, 3)
	gradExpected := make([]float64, 3)
	for i, v := range []float64{-2, 0, 2} {
		expected[i] = math.Log1p(math.Exp(v))
		gradExpected[i] = 1 / (1 + math.Exp(-v))
	}
	if !AlmostEqualSlices(out.Data(), expected, 1e-9) {
		t.Fatalf("softplus output mismatch: %v", out.Data())
	}
	if err := Sum(out).Backward(); err != nil {
		t.Fatalf("softplus backward failed: %v", err)
	}
	if !AlmostEqualSlices(input.Grad().Data(), gradExpected, 1e-9) {
		t.Fatalf("softplus grad mismatch: %v", input.Grad().Data())
	}
}

func TestGELUForwardBackward(t *testing.T) {
	input := MustNew([]float64{-1, 0, 1}, 3)
	input.SetRequiresGrad(true)
	out := GELU(input)
	invSqrt2 := 1 / math.Sqrt2
	expected := make([]float64, 3)
	gradExpected := make([]float64, 3)
	invSqrt2Pi := 1 / math.Sqrt(2*math.Pi)
	for i, v := range []float64{-1, 0, 1} {
		expected[i] = 0.5 * v * (1 + math.Erf(v*invSqrt2))
		erfTerm := math.Erf(v * invSqrt2)
		expTerm := math.Exp(-0.5 * v * v)
		gradExpected[i] = 0.5*(1+erfTerm) + v*expTerm*invSqrt2Pi
	}
	if !AlmostEqualSlices(out.Data(), expected, 1e-9) {
		t.Fatalf("gelu output mismatch: %v", out.Data())
	}
	if err := Sum(out).Backward(); err != nil {
		t.Fatalf("gelu backward failed: %v", err)
	}
	if !AlmostEqualSlices(input.Grad().Data(), gradExpected, 1e-9) {
		t.Fatalf("gelu grad mismatch: %v", input.Grad().Data())
	}
}
