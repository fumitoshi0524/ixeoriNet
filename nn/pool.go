package nn

import "github.com/fumitoshi0524/ixeoriNet/tensor"

type MaxPool2d struct {
	kernelH int
	kernelW int
	strideH int
	strideW int
	padH    int
	padW    int
}

func NewMaxPool2d(kernelH, kernelW, strideH, strideW, padH, padW int) *MaxPool2d {
	if strideH <= 0 {
		strideH = kernelH
	}
	if strideW <= 0 {
		strideW = kernelW
	}
	return &MaxPool2d{
		kernelH: kernelH,
		kernelW: kernelW,
		strideH: strideH,
		strideW: strideW,
		padH:    padH,
		padW:    padW,
	}
}

func (m *MaxPool2d) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.MaxPool2D(input, m.kernelH, m.kernelW, m.strideH, m.strideW, m.padH, m.padW)
}

func (m *MaxPool2d) Parameters() []*tensor.Tensor {
	return nil
}

func (m *MaxPool2d) ZeroGrad() {}

type AvgPool2d struct {
	kernelH int
	kernelW int
	strideH int
	strideW int
	padH    int
	padW    int
}

func NewAvgPool2d(kernelH, kernelW, strideH, strideW, padH, padW int) *AvgPool2d {
	if strideH <= 0 {
		strideH = kernelH
	}
	if strideW <= 0 {
		strideW = kernelW
	}
	return &AvgPool2d{
		kernelH: kernelH,
		kernelW: kernelW,
		strideH: strideH,
		strideW: strideW,
		padH:    padH,
		padW:    padW,
	}
}

func (a *AvgPool2d) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.AvgPool2D(input, a.kernelH, a.kernelW, a.strideH, a.strideW, a.padH, a.padW)
}

func (a *AvgPool2d) Parameters() []*tensor.Tensor {
	return nil
}

func (a *AvgPool2d) ZeroGrad() {}
