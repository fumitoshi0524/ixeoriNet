package nn

import (
	"math"
	"testing"

	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func TestSimpleRNNForwardBackward(t *testing.T) {
	rnn := NewSimpleRNN(1, 1, "tanh", true)
	mustSetData(t, rnn.weightIH, []float64{0.8})
	mustSetData(t, rnn.weightHH, []float64{0.1})
	mustSetData(t, rnn.biasIH, []float64{0.05})
	mustSetData(t, rnn.biasHH, []float64{-0.02})

	inputs := []float64{0.2, -0.1, 0.3}
	inputTensor := tensor.MustNew([]float64{0.2, -0.1, 0.3}, 3, 1, 1)
	inputTensor.SetRequiresGrad(true)

	output, hFinal, err := rnn.ForwardWithState(inputTensor, nil)
	if err != nil {
		t.Fatalf("simple rnn forward failed: %v", err)
	}

	expectedSeq, expectedLast := simpleRNNReference(inputs, 0.8, 0.1, 0.05, -0.02)
	if !floatsAlmostEqual(output.Data(), expectedSeq, 1e-6) {
		t.Fatalf("unexpected rnn outputs: got %v want %v", output.Data(), expectedSeq)
	}
	if math.Abs(hFinal.Data()[0]-expectedLast) > 1e-6 {
		t.Fatalf("unexpected final hidden: got %v want %v", hFinal.Data(), expectedLast)
	}

	sum := tensor.Sum(output)
	if err := sum.Backward(); err != nil {
		t.Fatalf("simple rnn backward failed: %v", err)
	}
	if inputGrad := inputTensor.Grad(); inputGrad == nil || inputGrad.Numel() != len(inputs) {
		t.Fatalf("missing input gradient")
	}
	for _, p := range rnn.Parameters() {
		if p != nil && p.Grad() == nil {
			t.Fatalf("parameter %p missing gradient", p)
		}
	}
}

func TestGRUForwardBackward(t *testing.T) {
	gru := NewGRU(1, 1, false)
	mustSetData(t, gru.weightIH[gruGateUpdate], []float64{0.15})
	mustSetData(t, gru.weightHH[gruGateUpdate], []float64{0.05})
	mustSetData(t, gru.weightIH[gruGateReset], []float64{-0.2})
	mustSetData(t, gru.weightHH[gruGateReset], []float64{0.1})
	mustSetData(t, gru.weightIH[gruGateNew], []float64{0.4})
	mustSetData(t, gru.weightHH[gruGateNew], []float64{0.3})

	inputs := []float64{0.2, -0.1, 0.3}
	inputTensor := tensor.MustNew([]float64{0.2, -0.1, 0.3}, 3, 1, 1)
	inputTensor.SetRequiresGrad(true)

	output, hFinal, err := gru.ForwardWithState(inputTensor, nil)
	if err != nil {
		t.Fatalf("gru forward failed: %v", err)
	}

	expectedSeq, expectedLast := gruReference(inputs, 0.15, 0.05, -0.2, 0.1, 0.4, 0.3)
	if !floatsAlmostEqual(output.Data(), expectedSeq, 1e-6) {
		t.Fatalf("unexpected gru outputs: got %v want %v", output.Data(), expectedSeq)
	}
	if math.Abs(hFinal.Data()[0]-expectedLast) > 1e-6 {
		t.Fatalf("unexpected gru final hidden: got %v want %v", hFinal.Data(), expectedLast)
	}

	sum := tensor.Sum(output)
	if err := sum.Backward(); err != nil {
		t.Fatalf("gru backward failed: %v", err)
	}
	if inputGrad := inputTensor.Grad(); inputGrad == nil || inputGrad.Numel() != len(inputs) {
		t.Fatalf("missing input gradient")
	}
	for _, p := range gru.Parameters() {
		if p != nil && p.Grad() == nil {
			t.Fatalf("parameter %p missing gradient", p)
		}
	}
}

func TestLSTMForwardBackward(t *testing.T) {
	lstm := NewLSTM(1, 1, false)
	mustSetData(t, lstm.weightIH[lstmGateInput], []float64{0.25})
	mustSetData(t, lstm.weightHH[lstmGateInput], []float64{0.1})
	mustSetData(t, lstm.weightIH[lstmGateForget], []float64{-0.3})
	mustSetData(t, lstm.weightHH[lstmGateForget], []float64{0.2})
	mustSetData(t, lstm.weightIH[lstmGateCell], []float64{0.45})
	mustSetData(t, lstm.weightHH[lstmGateCell], []float64{0.15})
	mustSetData(t, lstm.weightIH[lstmGateOutput], []float64{0.35})
	mustSetData(t, lstm.weightHH[lstmGateOutput], []float64{0.05})

	inputs := []float64{0.2, -0.1, 0.3}
	inputTensor := tensor.MustNew([]float64{0.2, -0.1, 0.3}, 3, 1, 1)
	inputTensor.SetRequiresGrad(true)

	output, hFinal, cFinal, err := lstm.ForwardWithState(inputTensor, nil, nil)
	if err != nil {
		t.Fatalf("lstm forward failed: %v", err)
	}

	expectedSeq, expectedH, expectedC := lstmReference(inputs,
		[lstmGateTotal]float64{0.25, -0.3, 0.45, 0.35},
		[lstmGateTotal]float64{0.1, 0.2, 0.15, 0.05},
	)
	if !floatsAlmostEqual(output.Data(), expectedSeq, 1e-6) {
		t.Fatalf("unexpected lstm outputs: got %v want %v", output.Data(), expectedSeq)
	}
	if math.Abs(hFinal.Data()[0]-expectedH) > 1e-6 {
		t.Fatalf("unexpected lstm final hidden: got %v want %v", hFinal.Data(), expectedH)
	}
	if math.Abs(cFinal.Data()[0]-expectedC) > 1e-6 {
		t.Fatalf("unexpected lstm final cell: got %v want %v", cFinal.Data(), expectedC)
	}

	sum := tensor.Sum(output)
	if err := sum.Backward(); err != nil {
		t.Fatalf("lstm backward failed: %v", err)
	}
	if inputGrad := inputTensor.Grad(); inputGrad == nil || inputGrad.Numel() != len(inputs) {
		t.Fatalf("missing input gradient")
	}
	for _, p := range lstm.Parameters() {
		if p != nil && p.Grad() == nil {
			t.Fatalf("parameter %p missing gradient", p)
		}
	}
}

func mustSetData(t *testing.T, tt *tensor.Tensor, vals []float64) {
	t.Helper()
	if tt == nil {
		if vals != nil {
			t.Fatalf("attempted to set data on nil tensor")
		}
		return
	}
	if err := tt.SetData(vals); err != nil {
		t.Fatalf("failed setting tensor data: %v", err)
	}
}

func simpleRNNReference(inputs []float64, wIH, wHH, bIH, bHH float64) ([]float64, float64) {
	outputs := make([]float64, len(inputs))
	h := 0.0
	for i, x := range inputs {
		sum := x*wIH + h*wHH + bIH + bHH
		h = math.Tanh(sum)
		outputs[i] = h
	}
	return outputs, h
}

func gruReference(inputs []float64, wIZ, wHZ, wIR, wHR, wIN, wHN float64) ([]float64, float64) {
	outputs := make([]float64, len(inputs))
	h := 0.0
	for i, x := range inputs {
		z := sigmoid(x*wIZ + h*wHZ)
		r := sigmoid(x*wIR + h*wHR)
		n := math.Tanh(x*wIN + (r*h)*wHN)
		h = (1-z)*n + z*h
		outputs[i] = h
	}
	return outputs, h
}

func lstmReference(inputs []float64, wIH, wHH [lstmGateTotal]float64) ([]float64, float64, float64) {
	outputs := make([]float64, len(inputs))
	h := 0.0
	c := 0.0
	for i, x := range inputs {
		iGate := sigmoid(x*wIH[lstmGateInput] + h*wHH[lstmGateInput])
		fGate := sigmoid(x*wIH[lstmGateForget] + h*wHH[lstmGateForget])
		gGate := math.Tanh(x*wIH[lstmGateCell] + h*wHH[lstmGateCell])
		oGate := sigmoid(x*wIH[lstmGateOutput] + h*wHH[lstmGateOutput])

		c = fGate*c + iGate*gGate
		h = oGate * math.Tanh(c)
		outputs[i] = h
	}
	return outputs, h, c
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
