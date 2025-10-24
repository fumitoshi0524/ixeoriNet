package nn

import (
	"errors"
	"fmt"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Module interface {
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)
	Parameters() []*tensor.Tensor
	ZeroGrad()
}

type StatefulModule interface {
	Module
	StateDict(prefix string, state map[string]*tensor.Tensor)
	LoadState(prefix string, state map[string]*tensor.Tensor) error
}

func ZeroGradAll(mods ...Module) {
	for _, m := range mods {
		if m == nil {
			continue
		}
		m.ZeroGrad()
	}
}

func SaveModule(path string, mod Module) error {
	if mod == nil {
		return errors.New("SaveModule requires non-nil module")
	}
	state := make(map[string]*tensor.Tensor)
	if sm, ok := mod.(StatefulModule); ok {
		sm.StateDict("", state)
	} else {
		captureParameters("", mod, state)
	}
	if len(state) == 0 {
		return errors.New("module has no state to save")
	}
	return tensor.SaveTensors(path, state)
}

func LoadModule(path string, mod Module) error {
	if mod == nil {
		return errors.New("LoadModule requires non-nil module")
	}
	state, err := tensor.LoadTensors(path)
	if err != nil {
		return err
	}
	if sm, ok := mod.(StatefulModule); ok {
		return sm.LoadState("", state)
	}
	return loadParameters("", mod, state)
}

func joinPrefix(prefix, name string) string {
	if prefix == "" {
		return name
	}
	if name == "" {
		return prefix
	}
	return prefix + "." + name
}

func captureParameters(prefix string, mod Module, state map[string]*tensor.Tensor) {
	params := mod.Parameters()
	for idx, p := range params {
		if p == nil {
			continue
		}
		key := joinPrefix(prefix, fmt.Sprintf("param_%d", idx))
		state[key] = p.Clone()
	}
}

func loadParameters(prefix string, mod Module, state map[string]*tensor.Tensor) error {
	params := mod.Parameters()
	for idx, p := range params {
		if p == nil {
			continue
		}
		key := joinPrefix(prefix, fmt.Sprintf("param_%d", idx))
		t, ok := state[key]
		if !ok {
			return fmt.Errorf("missing parameter %s", key)
		}
		if err := tensor.CopyInto(p, t); err != nil {
			return fmt.Errorf("load %s: %w", key, err)
		}
	}
	return nil
}
