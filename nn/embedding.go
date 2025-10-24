package nn

import (
	"fmt"
	"math"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Embedding struct {
	numEmbeddings int
	embeddingDim  int
	weight        *tensor.Tensor
}

func NewEmbedding(numEmbeddings, embeddingDim int) *Embedding {
	weight := tensor.Randn(numEmbeddings, embeddingDim)
	scale := 1.0 / math.Sqrt(float64(embeddingDim))
	weight.Scale(scale)
	weight.SetRequiresGrad(true)
	return &Embedding{
		numEmbeddings: numEmbeddings,
		embeddingDim:  embeddingDim,
		weight:        weight,
	}
}

func (e *Embedding) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	return tensor.Embedding(e.weight, input)
}

func (e *Embedding) Parameters() []*tensor.Tensor {
	return []*tensor.Tensor{e.weight}
}

func (e *Embedding) ZeroGrad() {
	e.weight.ZeroGrad()
}

func (e *Embedding) StateDict(prefix string, state map[string]*tensor.Tensor) {
	if state == nil {
		return
	}
	state[joinPrefix(prefix, "weight")] = e.weight.Clone()
}

func (e *Embedding) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	if state == nil {
		return fmt.Errorf("state dict is nil")
	}
	key := joinPrefix(prefix, "weight")
	w, ok := state[key]
	if !ok {
		return fmt.Errorf("Embedding missing %s", key)
	}
	if err := tensor.CopyInto(e.weight, w); err != nil {
		return fmt.Errorf("load %s: %w", key, err)
	}
	return nil
}
