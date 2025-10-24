package tensor

import (
	"errors"
)

type Tensor struct {
	data         []float64
	shape        []int
	strides      []int
	grad         *Tensor
	requiresGrad bool
	node         *node
	parents      []*Tensor
}

type node struct {
	backward func(grad *Tensor, grads map[*Tensor]*Tensor)
}

func New(data []float64, shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return nil, errors.New("shape is required")
	}
	total := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, errors.New("invalid shape")
		}
		total *= dim
	}
	if total != len(data) {
		return nil, errors.New("data and shape mismatch")
	}
	t := &Tensor{
		data:    append([]float64(nil), data...),
		shape:   append([]int(nil), shape...),
		strides: makeStrides(shape),
	}
	return t, nil
}

func MustNew(data []float64, shape ...int) *Tensor {
	t, err := New(data, shape...)
	if err != nil {
		panic(err)
	}
	return t
}

func Zeros(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return MustNew(make([]float64, size), shape...)
}

func Ones(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = 1
	}
	return MustNew(data, shape...)
}

func Full(value float64, shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	for i := range data {
		data[i] = value
	}
	return MustNew(data, shape...)
}

func (t *Tensor) Clone() *Tensor {
	if t == nil {
		return nil
	}
	clone := &Tensor{
		data:    append([]float64(nil), t.data...),
		shape:   append([]int(nil), t.shape...),
		strides: append([]int(nil), t.strides...),
	}
	return clone
}

func (t *Tensor) Shape() []int {
	return append([]int(nil), t.shape...)
}

func (t *Tensor) Numel() int {
	return len(t.data)
}

func (t *Tensor) Data() []float64 {
	return append([]float64(nil), t.data...)
}

// SetData overwrites the tensor's underlying values. The provided slice must match Numel().
func (t *Tensor) SetData(values []float64) error {
	if len(values) != len(t.data) {
		return errors.New("SetData expects matching element count")
	}
	copy(t.data, values)
	return nil
}

func (t *Tensor) SetRequiresGrad(v bool) {
	t.requiresGrad = v
}

func (t *Tensor) RequiresGrad() bool {
	return t.requiresGrad
}

func (t *Tensor) Grad() *Tensor {
	if t.grad == nil {
		return nil
	}
	return t.grad.Clone()
}

func (t *Tensor) ZeroGrad() {
	t.grad = nil
}

func (t *Tensor) Detach() *Tensor {
	clone := t.Clone()
	clone.requiresGrad = false
	clone.node = nil
	clone.parents = nil
	return clone
}

// CopyInto copies the contents of src into dst, ensuring shapes match.
func CopyInto(dst, src *Tensor) error {
	if dst == nil || src == nil {
		return errors.New("CopyInto requires non-nil tensors")
	}
	if len(dst.shape) != len(src.shape) {
		return errors.New("CopyInto shape rank mismatch")
	}
	for i, dim := range dst.shape {
		if dim != src.shape[i] {
			return errors.New("CopyInto shape mismatch")
		}
	}
	if len(dst.data) != len(src.data) {
		return errors.New("CopyInto data size mismatch")
	}
	copy(dst.data, src.data)
	return nil
}

func makeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}
