package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/fumitoshi0524/ixeoriNet/loss"
	"github.com/fumitoshi0524/ixeoriNet/nn"
	"github.com/fumitoshi0524/ixeoriNet/optim"
	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	classes := 3
	samplesPerClass := 100
	samples := classes * samplesPerClass
	features := 2
	inputs := make([]float64, samples*features)
	targets := make([]int, samples)
	for c := 0; c < classes; c++ {
		angle := float64(c) * 2 * math.Pi / float64(classes)
		centerX := math.Cos(angle) * 2
		centerY := math.Sin(angle) * 2
		for i := 0; i < samplesPerClass; i++ {
			idx := c*samplesPerClass + i
			x := centerX + rand.NormFloat64()*0.5
			y := centerY + rand.NormFloat64()*0.5
			inputs[idx*features+0] = x
			inputs[idx*features+1] = y
			targets[idx] = c
		}
	}
	xs := tensor.MustNew(inputs, samples, features)
	model := nn.NewSequential(
		nn.NewLinear(features, 32, true),
		nn.Relu(),
		nn.NewLinear(32, classes, true),
	)
	opt := optim.NewAdam(model.Parameters(), 0.01, 0.9, 0.999, 1e-8)
	epochs := 300
	for epoch := 0; epoch < epochs; epoch++ {
		opt.ZeroGrad()
		pred, err := model.Forward(xs)
		if err != nil {
			panic(err)
		}
		lossVal, err := loss.CrossEntropy(pred, targets)
		if err != nil {
			panic(err)
		}
		if err := lossVal.Backward(); err != nil {
			panic(err)
		}
		if err := opt.Step(); err != nil {
			panic(err)
		}
		if epoch%50 == 0 || epoch == epochs-1 {
			acc := accuracy(pred.Data(), targets, classes)
			fmt.Printf("epoch %d loss %.4f acc %.2f%%\n", epoch, lossVal.Data()[0], acc*100)
		}
	}
}

func accuracy(scores []float64, targets []int, classes int) float64 {
	samples := len(targets)
	correct := 0
	for i := 0; i < samples; i++ {
		bestIdx := 0
		bestVal := scores[i*classes]
		for j := 1; j < classes; j++ {
			v := scores[i*classes+j]
			if v > bestVal {
				bestVal = v
				bestIdx = j
			}
		}
		if bestIdx == targets[i] {
			correct++
		}
	}
	return float64(correct) / float64(samples)
}
