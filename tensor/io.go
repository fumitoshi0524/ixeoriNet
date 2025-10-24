package tensor

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
)

type tensorRecord struct {
	Shape []int     `json:"shape"`
	Data  []float64 `json:"data"`
}

// SaveTensors serializes a named tensor set to disk using JSON.
func SaveTensors(path string, tensors map[string]*Tensor) error {
	if len(tensors) == 0 {
		return errors.New("SaveTensors requires at least one tensor")
	}
	records := make(map[string]tensorRecord, len(tensors))
	for name, t := range tensors {
		if t == nil {
			return fmt.Errorf("tensor %s is nil", name)
		}
		records[name] = tensorRecord{Shape: t.Shape(), Data: t.Data()}
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(records)
}

// LoadTensors deserializes tensors saved with SaveTensors.
func LoadTensors(path string) (map[string]*Tensor, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	records := make(map[string]tensorRecord)
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&records); err != nil {
		return nil, err
	}
	result := make(map[string]*Tensor, len(records))
	for name, rec := range records {
		if len(rec.Shape) == 0 {
			return nil, fmt.Errorf("tensor %s missing shape", name)
		}
		t, err := New(rec.Data, rec.Shape...)
		if err != nil {
			return nil, fmt.Errorf("tensor %s: %w", name, err)
		}
		result[name] = t
	}
	return result, nil
}
