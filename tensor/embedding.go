package tensor

import "errors"

// Embedding looks up embeddings for given indices from weight matrix.
// weight shape: [num_embeddings, embedding_dim...]
// index shape: arbitrary; values are treated as integer indices.
func Embedding(weight *Tensor, index *Tensor) (*Tensor, error) {
	if index == nil {
		return nil, errors.New("index tensor required")
	}
	if len(weight.shape) < 2 {
		return nil, errors.New("weight must have rank >= 2")
	}
	numEmb := weight.shape[0]
	embedSize := 1
	for _, dim := range weight.shape[1:] {
		embedSize *= dim
	}
	outShape := append([]int(nil), index.shape...)
	outShape = append(outShape, weight.shape[1:]...)
	out := Zeros(outShape...)

	totalIndices := len(index.data)
	for idx := 0; idx < totalIndices; idx++ {
		val := int(index.data[idx])
		if val < 0 || val >= numEmb {
			return nil, errors.New("embedding index out of range")
		}
		srcStart := val * embedSize
		dstStart := idx * embedSize
		copy(out.data[dstStart:dstStart+embedSize], weight.data[srcStart:srcStart+embedSize])
	}

	if !weight.requiresGrad {
		return out, nil
	}

	out.requiresGrad = true
	out.parents = []*Tensor{weight}
	out.node = &node{
		backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
			gWeight := Zeros(weight.shape...)
			for idx := 0; idx < totalIndices; idx++ {
				val := int(index.data[idx])
				srcStart := idx * embedSize
				dstStart := val * embedSize
				for j := 0; j < embedSize; j++ {
					gWeight.data[dstStart+j] += grad.data[srcStart+j]
				}
			}
			accumulate(grads, weight, gWeight)
		},
	}

	return out, nil
}
