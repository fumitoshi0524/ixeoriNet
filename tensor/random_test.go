package tensor

import (
	"math"
	"testing"
)

func TestRandnShapeAndStats(t *testing.T) {
	samples := Randn(1000, 10)
	if !equalShapes(samples.Shape(), []int{1000, 10}) {
		t.Fatalf("unexpected shape: %v", samples.Shape())
	}
	data := samples.Data()
	n := float64(len(data))
	if n == 0 {
		t.Fatalf("randn returned empty data")
	}
	mean := 0.0
	for _, v := range data {
		mean += v
	}
	mean /= n
	variance := 0.0
	for _, v := range data {
		diff := v - mean
		variance += diff * diff
	}
	variance /= n

	if math.Abs(mean) > 0.1 {
		t.Fatalf("randn mean too far from zero: %.6f", mean)
	}
	if variance < 0.8 || variance > 1.2 {
		t.Fatalf("randn variance unexpected: %.6f", variance)
	}
}

func TestRandnProducesDifferentSamples(t *testing.T) {
	a := Randn(5, 5).Data()
	b := Randn(5, 5).Data()
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatalf("consecutive Randn calls produced identical samples")
	}
}
