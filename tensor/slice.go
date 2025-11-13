package tensor

import (
    "errors"
    "github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

// SliceRows2D returns a view of consecutive rows [rowStart, rowStart+rows) of a rank-2 tensor.
// The returned tensor shares the underlying data slice and supports autograd (accumulates
// gradients back to the source tensor in the corresponding region).
func SliceRows2D(t *Tensor, rowStart, rows int) (*Tensor, error) {
    if t == nil {
        return nil, errors.New("nil tensor")
    }
    if len(t.shape) != 2 {
        return nil, errors.New("SliceRows2D expects rank-2 tensor")
    }
    cols := t.shape[1]
    if rowStart < 0 || rows < 0 || rowStart+rows > t.shape[0] {
        return nil, errors.New("slice out of range")
    }
    start := rowStart * cols
    end := (rowStart + rows) * cols
    out := &Tensor{
        data:    t.data[start:end],
        shape:   []int{rows, cols},
        strides: makeStrides([]int{rows, cols}),
        // preserve requiresGrad so autograd is wired
        requiresGrad: t.requiresGrad,
    }
    if t.requiresGrad {
        out.parents = []*Tensor{t}
        // capture copies used in closure
        rs := rowStart
        rrows := rows
        colsCopy := cols
        out.node = &node{
            backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
                g := Zeros(t.shape...)
                parallel.For(rrows, func(startRow, endRow int) {
                    for r := startRow; r < endRow; r++ {
                        dstBase := (rs + r) * colsCopy
                        srcBase := r * colsCopy
                        for c := 0; c < colsCopy; c++ {
                            g.data[dstBase+c] += grad.data[srcBase+c]
                        }
                    }
                })
                accumulate(grads, t, g)
            },
        }
    }
    return out, nil
}
