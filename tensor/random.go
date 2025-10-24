package tensor

import (
	"math/rand"
	"sync"
	"time"
)

var rng = rand.New(rand.NewSource(time.Now().UnixNano()))
var rngLock sync.Mutex

func Randn(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	rngLock.Lock()
	for i := range data {
		data[i] = rng.NormFloat64()
	}
	rngLock.Unlock()
	return MustNew(data, shape...)
}
