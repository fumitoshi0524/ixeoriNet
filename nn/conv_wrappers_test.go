package nn

import (
	"testing"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func TestConv1dWrapperMatchesTensor(t *testing.T) {
	conv := NewConv1d(1, 1, 3, 1, 0, true)
	if err := conv.weight.SetData([]float64{1, 0, -1}); err != nil {
		t.Fatalf("set weight: %v", err)
	}
	if err := conv.bias.SetData([]float64{0.5}); err != nil {
		t.Fatalf("set bias: %v", err)
	}
	input := tensor.MustNew([]float64{1, 2, 3, 4, 5}, 1, 1, 5)
	input.SetRequiresGrad(true)

	out, err := conv.Forward(input)
	if err != nil {
		t.Fatalf("conv1d forward failed: %v", err)
	}

	refInput := input.Detach()
	refWeight := conv.weight.Detach()
	var refBias *tensor.Tensor
	if conv.bias != nil {
		refBias = conv.bias.Detach()
	}
	ref, err := tensor.Conv1D(refInput, refWeight, refBias, conv.stride, conv.pad)
	if err != nil {
		t.Fatalf("reference conv1d failed: %v", err)
	}
	if !floatsAlmostEqual(out.Data(), ref.Data(), 1e-9) {
		t.Fatalf("conv1d wrapper mismatch: got %v want %v", out.Data(), ref.Data())
	}

	loss := tensor.Sum(out)
	if err := loss.Backward(); err != nil {
		t.Fatalf("conv1d backward failed: %v", err)
	}
	if conv.weight.Grad() == nil {
		t.Fatalf("expected gradient on conv1d weight")
	}
	if conv.bias.Grad() == nil {
		t.Fatalf("expected gradient on conv1d bias")
	}
}

func TestConv2dWrapperMatchesTensor(t *testing.T) {
	conv := NewConv2d(1, 1, 2, 2, 1, 1, 0, 0, true)
	if err := conv.weight.SetData([]float64{1, -1, 2, 0}); err != nil {
		t.Fatalf("set weight: %v", err)
	}
	if err := conv.bias.SetData([]float64{-0.25}); err != nil {
		t.Fatalf("set bias: %v", err)
	}
	input := tensor.MustNew([]float64{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}, 1, 1, 3, 3)
	input.SetRequiresGrad(true)

	out, err := conv.Forward(input)
	if err != nil {
		t.Fatalf("conv2d forward failed: %v", err)
	}

	ref, err := tensor.Conv2D(input.Detach(), conv.weight.Detach(), conv.bias.Detach(), conv.strideH, conv.strideW, conv.padH, conv.padW)
	if err != nil {
		t.Fatalf("reference conv2d failed: %v", err)
	}
	if !floatsAlmostEqual(out.Data(), ref.Data(), 1e-9) {
		t.Fatalf("conv2d wrapper mismatch: got %v want %v", out.Data(), ref.Data())
	}

	loss := tensor.Sum(out)
	if err := loss.Backward(); err != nil {
		t.Fatalf("conv2d backward failed: %v", err)
	}
	if conv.weight.Grad() == nil {
		t.Fatalf("expected gradient on conv2d weight")
	}
	if conv.bias.Grad() == nil {
		t.Fatalf("expected gradient on conv2d bias")
	}
}

func TestConv3dWrapperMatchesTensor(t *testing.T) {
	conv := NewConv3d(1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, false)
	if err := conv.weight.SetData([]float64{
		1, 0,
		-1, 1,
		2, -2,
		0.5, -0.5,
	}); err != nil {
		t.Fatalf("set weight: %v", err)
	}
	input := tensor.MustNew([]float64{
		1, 2,
		3, 4,

		5, 6,
		7, 8,
	}, 1, 1, 2, 2, 2)
	input.SetRequiresGrad(true)

	out, err := conv.Forward(input)
	if err != nil {
		t.Fatalf("conv3d forward failed: %v", err)
	}

	ref, err := tensor.Conv3D(input.Detach(), conv.weight.Detach(), nil, conv.strideD, conv.strideH, conv.strideW, conv.padD, conv.padH, conv.padW)
	if err != nil {
		t.Fatalf("reference conv3d failed: %v", err)
	}
	if !floatsAlmostEqual(out.Data(), ref.Data(), 1e-9) {
		t.Fatalf("conv3d wrapper mismatch: got %v want %v", out.Data(), ref.Data())
	}

	loss := tensor.Sum(out)
	if err := loss.Backward(); err != nil {
		t.Fatalf("conv3d backward failed: %v", err)
	}
	if conv.weight.Grad() == nil {
		t.Fatalf("expected gradient on conv3d weight")
	}
}

func TestConvTransposeWrappersMatchTensor(t *testing.T) {
	ct1d := NewConvTranspose1d(1, 1, 2, 1, 0, false)
	if err := ct1d.weight.SetData([]float64{1, -1}); err != nil {
		t.Fatalf("set transposed conv1d weight: %v", err)
	}
	input1d := tensor.MustNew([]float64{1, 2, 3}, 1, 1, 3)
	input1d.SetRequiresGrad(true)
	out1d, err := ct1d.Forward(input1d)
	if err != nil {
		t.Fatalf("convtranspose1d forward failed: %v", err)
	}
	ref1d, err := tensor.ConvTranspose1D(input1d.Detach(), ct1d.weight.Detach(), nil, ct1d.stride, ct1d.padding)
	if err != nil {
		t.Fatalf("reference convtranspose1d failed: %v", err)
	}
	if !floatsAlmostEqual(out1d.Data(), ref1d.Data(), 1e-9) {
		t.Fatalf("convtranspose1d mismatch")
	}

	loss1d := tensor.Sum(out1d)
	if err := loss1d.Backward(); err != nil {
		t.Fatalf("convtranspose1d backward failed: %v", err)
	}
	if ct1d.weight.Grad() == nil {
		t.Fatalf("expected gradient on convtranspose1d weight")
	}

	ct2d := NewConvTranspose2d(1, 1, 2, 2, 1, 1, 0, 0, true)
	if err := ct2d.weight.SetData([]float64{1, 0, -1, 1}); err != nil {
		t.Fatalf("set transposed conv2d weight: %v", err)
	}
	if err := ct2d.bias.SetData([]float64{0.1}); err != nil {
		t.Fatalf("set transposed conv2d bias: %v", err)
	}
	input2d := tensor.MustNew([]float64{1, 2, 3, 4}, 1, 1, 2, 2)
	input2d.SetRequiresGrad(true)
	out2d, err := ct2d.Forward(input2d)
	if err != nil {
		t.Fatalf("convtranspose2d forward failed: %v", err)
	}
	ref2d, err := tensor.ConvTranspose2D(input2d.Detach(), ct2d.weight.Detach(), ct2d.bias.Detach(), ct2d.strideH, ct2d.strideW, ct2d.padH, ct2d.padW)
	if err != nil {
		t.Fatalf("reference convtranspose2d failed: %v", err)
	}
	if !floatsAlmostEqual(out2d.Data(), ref2d.Data(), 1e-9) {
		t.Fatalf("convtranspose2d mismatch: got %v want %v", out2d.Data(), ref2d.Data())
	}

	loss2d := tensor.Sum(out2d)
	if err := loss2d.Backward(); err != nil {
		t.Fatalf("convtranspose2d backward failed: %v", err)
	}
	if ct2d.weight.Grad() == nil {
		t.Fatalf("expected gradient on convtranspose2d weight")
	}
	if ct2d.bias.Grad() == nil {
		t.Fatalf("expected gradient on convtranspose2d bias")
	}

	ct3d := NewConvTranspose3d(1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, false)
	if err := ct3d.weight.SetData([]float64{
		1, 0,
		0, 1,
		-1, 1,
		2, -2,
	}); err != nil {
		t.Fatalf("set transposed conv3d weight: %v", err)
	}
	input3d := tensor.MustNew([]float64{
		1, 1,
		1, 1,

		2, 2,
		2, 2,
	}, 1, 1, 2, 2, 2)
	input3d.SetRequiresGrad(true)
	out3d, err := ct3d.Forward(input3d)
	if err != nil {
		t.Fatalf("convtranspose3d forward failed: %v", err)
	}
	ref3d, err := tensor.ConvTranspose3D(input3d.Detach(), ct3d.weight.Detach(), nil, ct3d.strideD, ct3d.strideH, ct3d.strideW, ct3d.padD, ct3d.padH, ct3d.padW)
	if err != nil {
		t.Fatalf("reference convtranspose3d failed: %v", err)
	}
	if !floatsAlmostEqual(out3d.Data(), ref3d.Data(), 1e-9) {
		t.Fatalf("convtranspose3d mismatch: got %v want %v", out3d.Data(), ref3d.Data())
	}

	loss3d := tensor.Sum(out3d)
	if err := loss3d.Backward(); err != nil {
		t.Fatalf("convtranspose3d backward failed: %v", err)
	}
	if ct3d.weight.Grad() == nil {
		t.Fatalf("expected gradient on convtranspose3d weight")
	}
}

func TestConv1dStateDictRoundTrip(t *testing.T) {
	conv := NewConv1d(1, 1, 2, 1, 0, true)
	if err := conv.weight.SetData([]float64{0.25, -0.75}); err != nil {
		t.Fatalf("set weight: %v", err)
	}
	if err := conv.bias.SetData([]float64{0.1}); err != nil {
		t.Fatalf("set bias: %v", err)
	}
	state := map[string]*tensor.Tensor{}
	conv.StateDict("layer", state)
	clone := NewConv1d(1, 1, 2, 1, 0, true)
	if err := clone.LoadState("layer", state); err != nil {
		t.Fatalf("LoadState failed: %v", err)
	}
	if !floatsAlmostEqual(conv.weight.Data(), clone.weight.Data(), 1e-9) {
		t.Fatalf("weight mismatch after LoadState")
	}
	if !floatsAlmostEqual(conv.bias.Data(), clone.bias.Data(), 1e-9) {
		t.Fatalf("bias mismatch after LoadState")
	}
}
