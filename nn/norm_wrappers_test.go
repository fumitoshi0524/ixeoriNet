package nn

import (
	"testing"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func TestLayerNormWrapperMatchesTensor(t *testing.T) {
	ln := NewLayerNorm([]int{3}, 1e-5, true)
	if err := ln.weight.SetData([]float64{1.0, 0.5, -0.5}); err != nil {
		t.Fatalf("set layernorm weight: %v", err)
	}
	if err := ln.bias.SetData([]float64{0.1, -0.2, 0.3}); err != nil {
		t.Fatalf("set layernorm bias: %v", err)
	}
	input := tensor.MustNew([]float64{
		1, 2, 3,
		4, 5, 6,
	}, 2, 3)
	input.SetRequiresGrad(true)

	out, err := ln.Forward(input)
	if err != nil {
		t.Fatalf("layernorm forward failed: %v", err)
	}

	ref, err := tensor.LayerNorm(input.Detach(), ln.normalizedShape, ln.weight.Detach(), ln.bias.Detach(), ln.eps)
	if err != nil {
		t.Fatalf("reference layernorm failed: %v", err)
	}
	if !floatsAlmostEqual(out.Data(), ref.Data(), 1e-9) {
		t.Fatalf("layernorm wrapper mismatch: got %v want %v", out.Data(), ref.Data())
	}

	loss := tensor.Sum(out)
	if err := loss.Backward(); err != nil {
		t.Fatalf("layernorm backward failed: %v", err)
	}
	if ln.weight.Grad() == nil || ln.bias.Grad() == nil {
		t.Fatalf("expected gradients on layernorm affine params")
	}
}

func TestBatchNormWrapperMatchesTensorAndUpdatesState(t *testing.T) {
	bn := NewBatchNorm(2, 0.2, 1e-5, true)
	if err := bn.weight.SetData([]float64{1.0, 0.5}); err != nil {
		t.Fatalf("set batchnorm weight: %v", err)
	}
	if err := bn.bias.SetData([]float64{0.0, 0.1}); err != nil {
		t.Fatalf("set batchnorm bias: %v", err)
	}
	input := tensor.MustNew([]float64{
		1, 2,
		3, 4,
	}, 2, 2)
	input.SetRequiresGrad(true)

	meanClone := bn.runningMean.Clone()
	varClone := bn.runningVar.Clone()
	ref, err := tensor.BatchNorm(input.Detach(), meanClone, varClone, bn.weight.Detach(), bn.bias.Detach(), bn.momentum, bn.eps, true)
	if err != nil {
		t.Fatalf("reference batchnorm failed: %v", err)
	}

	out, err := bn.Forward(input)
	if err != nil {
		t.Fatalf("batchnorm forward failed: %v", err)
	}
	if !floatsAlmostEqual(out.Data(), ref.Data(), 1e-9) {
		t.Fatalf("batchnorm wrapper mismatch: got %v want %v", out.Data(), ref.Data())
	}
	if !floatsAlmostEqual(bn.runningMean.Data(), meanClone.Data(), 1e-9) {
		t.Fatalf("running mean mismatch: got %v want %v", bn.runningMean.Data(), meanClone.Data())
	}
	if !floatsAlmostEqual(bn.runningVar.Data(), varClone.Data(), 1e-9) {
		t.Fatalf("running var mismatch: got %v want %v", bn.runningVar.Data(), varClone.Data())
	}

	loss := tensor.Sum(out)
	if err := loss.Backward(); err != nil {
		t.Fatalf("batchnorm backward failed: %v", err)
	}
	if bn.weight.Grad() == nil || bn.bias.Grad() == nil {
		t.Fatalf("expected gradients on batchnorm affine params")
	}

	bn.Eval()
	evalInput := tensor.MustNew([]float64{
		-1, 0,
		1, 2,
	}, 2, 2)
	eval, err := bn.Forward(evalInput)
	if err != nil {
		t.Fatalf("batchnorm eval forward failed: %v", err)
	}
	refEval, err := tensor.BatchNorm(evalInput.Detach(), bn.runningMean.Clone(), bn.runningVar.Clone(), bn.weight.Detach(), bn.bias.Detach(), bn.momentum, bn.eps, false)
	if err != nil {
		t.Fatalf("reference eval batchnorm failed: %v", err)
	}
	if !floatsAlmostEqual(eval.Data(), refEval.Data(), 1e-9) {
		t.Fatalf("batchnorm eval mismatch: got %v want %v", eval.Data(), refEval.Data())
	}
}
