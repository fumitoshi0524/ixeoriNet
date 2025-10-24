package parallel

import (
	"sync/atomic"
	"testing"
)

func TestForCoversEntireRange(t *testing.T) {
	n := 37
	counts := make([]int32, n)
	For(n, func(start, end int) {
		for i := start; i < end; i++ {
			atomic.AddInt32(&counts[i], 1)
		}
	})
	for i, c := range counts {
		if c != 1 {
			t.Fatalf("expected index %d to be processed once, got %d", i, c)
		}
	}
}

func TestForNoopOnNonPositive(t *testing.T) {
	called := false
	For(0, func(start, end int) {
		called = true
	})
	if called {
		t.Fatalf("expected callback to remain unused")
	}
}
