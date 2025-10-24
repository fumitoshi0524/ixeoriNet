package tensor

import "errors"

// Gather selects values along axis according to integer indices.
func Gather(input *Tensor, axis int, index *Tensor) (*Tensor, error) {
	if index == nil {
		return nil, errors.New("index tensor required")
	}
	rank := len(input.shape)
	if rank == 0 {
		return nil, errors.New("gather requires rank >= 1 input")
	}
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return nil, errors.New("axis out of range")
	}
	if len(index.shape) != rank {
		return nil, errors.New("index rank mismatch")
	}
	for i, dim := range index.shape {
		if i == axis {
			continue
		}
		if dim != input.shape[i] {
			return nil, errors.New("index shape mismatch")
		}
	}

	out := Zeros(index.shape...)
	axisSize := input.shape[axis]
	indexAxis := index.shape[axis]
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= input.shape[i]
	}
	inner := 1
	for i := axis + 1; i < rank; i++ {
		inner *= input.shape[i]
	}

	for o := 0; o < outer; o++ {
		for ia := 0; ia < indexAxis; ia++ {
			for inr := 0; inr < inner; inr++ {
				idxOffset := ((o*indexAxis)+ia)*inner + inr
				idxVal := int(index.data[idxOffset])
				if idxVal < 0 || idxVal >= axisSize {
					return nil, errors.New("gather index out of range")
				}
				inOffset := ((o*axisSize)+idxVal)*inner + inr
				out.data[idxOffset] = input.data[inOffset]
			}
		}
	}

	if input.requiresGrad {
		out.requiresGrad = true
		out.parents = []*Tensor{input}
		out.node = &node{
			backward: func(grad *Tensor, grads map[*Tensor]*Tensor) {
				gInput := Zeros(input.shape...)
				for o := 0; o < outer; o++ {
					for ia := 0; ia < indexAxis; ia++ {
						for inr := 0; inr < inner; inr++ {
							idxOffset := ((o*indexAxis)+ia)*inner + inr
							idxVal := int(index.data[idxOffset])
							inOffset := ((o*axisSize)+idxVal)*inner + inr
							gInput.data[inOffset] += grad.data[idxOffset]
						}
					}
				}
				accumulate(grads, input, gInput)
			},
		}
	}

	return out, nil
}
