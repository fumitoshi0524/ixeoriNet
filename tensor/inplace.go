package tensor

import "github.com/fumitoshi0524/ixeoriNet/internal/parallel"

func (t *Tensor) Scale(v float64) {
	parallel.For(len(t.data), func(start, end int) {
		for i := start; i < end; i++ {
			t.data[i] *= v
		}
	})
}

func (t *Tensor) AddScaled(other *Tensor, alpha float64) error {
	if err := ensureSameShape(t, other); err != nil {
		return err
	}
	parallel.For(len(t.data), func(start, end int) {
		for i := start; i < end; i++ {
			t.data[i] += alpha * other.data[i]
		}
	})
	return nil
}

func (t *Tensor) MulInPlace(other *Tensor) error {
	if err := ensureSameShape(t, other); err != nil {
		return err
	}
	parallel.For(len(t.data), func(start, end int) {
		for i := start; i < end; i++ {
			t.data[i] *= other.data[i]
		}
	})
	return nil
}
