package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/fumitoshi0524/ixeoriNet/loss"
	"github.com/fumitoshi0524/ixeoriNet/nn"
	"github.com/fumitoshi0524/ixeoriNet/optim"
	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	samples := 256
	inputs := make([]float64, samples)
	targets := make([]float64, samples)
	for i := 0; i < samples; i++ {
		x := rand.Float64()*10 - 5
		y := 3.5*x - 1.2
		inputs[i] = x
		targets[i] = y
	}
	xs := tensor.MustNew(inputs, samples, 1)
	ys := tensor.MustNew(targets, samples, 1)
	linear := nn.NewLinear(1, 1, true)
	model := nn.NewSequential(linear)
	opt := optim.NewSGD(model.Parameters(), 0.01, 0.9)
	epochs := 200
	for epoch := 0; epoch < epochs; epoch++ {
		opt.ZeroGrad()
		pred, err := model.Forward(xs)
		if err != nil {
			panic(err)
		}
		lossVal, err := loss.MSE(pred, ys)
		if err != nil {
			panic(err)
		}
		if err := lossVal.Backward(); err != nil {
			panic(err)
		}
		if err := opt.Step(); err != nil {
			panic(err)
		}
		if epoch%20 == 0 || epoch == epochs-1 {
			fmt.Printf("epoch %d loss %.4f\n", epoch, lossVal.Data()[0])
		}
	}
	fmt.Printf("weight %.3f bias %.3f\n", linear.Weight().Data()[0], linear.Bias().Data()[0])
}
