package tensor

import (
	"errors"

	"github.com/fumitoshi0524/ixeoriNet/internal/parallel"
)

func (t *Tensor) Backward() error {
	if t == nil {
		return errors.New("nil tensor")
	}
	if !t.requiresGrad {
		return errors.New("tensor does not require grad")
	}
	order := topo(t)
	grads := map[*Tensor]*Tensor{}
	grads[t] = Full(1, t.shape...)
	for i := len(order) - 1; i >= 0; i-- {
		current := order[i]
		grad := grads[current]
		if grad == nil {
			continue
		}
		if current.grad == nil {
			current.grad = grad.Clone()
		} else {
			addInPlace(current.grad, grad)
		}
		if current.node != nil {
			current.node.backward(grad, grads)
		}
	}
	return nil
}

func topo(root *Tensor) []*Tensor {
	visited := map[*Tensor]bool{}
	var order []*Tensor
	var visit func(*Tensor)
	visit = func(node *Tensor) {
		if node == nil {
			return
		}
		if visited[node] {
			return
		}
		visited[node] = true
		for _, parent := range node.parents {
			visit(parent)
		}
		order = append(order, node)
	}
	visit(root)
	return order
}

func accumulate(grads map[*Tensor]*Tensor, target *Tensor, value *Tensor) {
	if target == nil || value == nil {
		return
	}
	if existing, ok := grads[target]; ok {
		addInPlace(existing, value)
	} else {
		grads[target] = value.Clone()
	}
}

func addInPlace(dst, src *Tensor) {
	if err := ensureSameShape(dst, src); err != nil {
		panic(err)
	}
	parallel.For(len(dst.data), func(start, end int) {
		for i := start; i < end; i++ {
			dst.data[i] += src.data[i]
		}
	})
}
