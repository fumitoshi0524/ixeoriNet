package nn

import "github.com/fumitoshi0524/ixeoriNet/tensor"

// FuncModule wraps a function into a Module.
type FuncModule struct {
    fn func(*tensor.Tensor) (*tensor.Tensor, error)
}

func NewModuleFunc(fn func(*tensor.Tensor) (*tensor.Tensor, error)) *FuncModule {
    return &FuncModule{fn: fn}
}

func (f *FuncModule) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
    return f.fn(input)
}

func (f *FuncModule) Parameters() []*tensor.Tensor { return nil }

func (f *FuncModule) ZeroGrad() {}
