package tensor

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveAndLoadTensors(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "state.json")

	tensors := map[string]*Tensor{
		"a": MustNew([]float64{1, 2, 3, 4}, 2, 2),
		"b": MustNew([]float64{0.5, -0.5}, 2),
	}
	if err := SaveTensors(path, tensors); err != nil {
		t.Fatalf("SaveTensors failed: %v", err)
	}
	loaded, err := LoadTensors(path)
	if err != nil {
		t.Fatalf("LoadTensors failed: %v", err)
	}
	if len(loaded) != len(tensors) {
		t.Fatalf("expected %d tensors, got %d", len(tensors), len(loaded))
	}
	for name, original := range tensors {
		loadedTensor, ok := loaded[name]
		if !ok {
			t.Fatalf("missing tensor %s", name)
		}
		if !AlmostEqualSlices(original.Data(), loadedTensor.Data(), 1e-9) {
			t.Fatalf("tensor %s data mismatch", name)
		}
		if !equalShapes(original.Shape(), loadedTensor.Shape()) {
			t.Fatalf("tensor %s shape mismatch", name)
		}
	}
}

func TestSaveTensorsValidatesInput(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "empty.json")
	if err := SaveTensors(path, map[string]*Tensor{}); err == nil {
		t.Fatalf("expected error for empty tensor map")
	}

	path = filepath.Join(tmpDir, "nil.json")
	tensors := map[string]*Tensor{"nil": nil}
	if err := SaveTensors(path, tensors); err == nil {
		t.Fatalf("expected error when tensor is nil")
	}
}

func equalShapes(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func AlmostEqualSlices(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}

func TestLoadTensorsMissingFile(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "missing.json")
	if _, err := LoadTensors(path); err == nil {
		t.Fatalf("expected error for missing file")
	}
}

func TestLoadTensorsMalformed(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "bad.json")
	if err := os.WriteFile(path, []byte("not json"), 0600); err != nil {
		t.Fatalf("write file: %v", err)
	}
	if _, err := LoadTensors(path); err == nil {
		t.Fatalf("expected error for malformed json")
	}
}
