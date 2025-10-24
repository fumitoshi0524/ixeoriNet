package nn

import (
	"fmt"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

type Sequential struct {
	modules []Module
}

func NewSequential(mods ...Module) *Sequential {
	copyMods := make([]Module, len(mods))
	copy(copyMods, mods)
	return &Sequential{modules: copyMods}
}

func (s *Sequential) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	var err error
	out := input
	for _, m := range s.modules {
		out, err = m.Forward(out)
		if err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (s *Sequential) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	for _, m := range s.modules {
		params = append(params, m.Parameters()...)
	}
	return params
}

func (s *Sequential) ZeroGrad() {
	for _, m := range s.modules {
		m.ZeroGrad()
	}
}

func (s *Sequential) StateDict(prefix string, state map[string]*tensor.Tensor) {
	for idx, mod := range s.modules {
		childPrefix := joinPrefix(prefix, fmt.Sprintf("%d", idx))
		if sm, ok := mod.(StatefulModule); ok {
			sm.StateDict(childPrefix, state)
		} else if len(mod.Parameters()) > 0 {
			captureParameters(childPrefix, mod, state)
		}
	}
}

func (s *Sequential) LoadState(prefix string, state map[string]*tensor.Tensor) error {
	for idx, mod := range s.modules {
		childPrefix := joinPrefix(prefix, fmt.Sprintf("%d", idx))
		if sm, ok := mod.(StatefulModule); ok {
			if err := sm.LoadState(childPrefix, state); err != nil {
				return err
			}
		} else if len(mod.Parameters()) > 0 {
			if err := loadParameters(childPrefix, mod, state); err != nil {
				return err
			}
		}
	}
	return nil
}
