package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	mnistdata "github.com/fumitoshi0524/ixeoriNet/examples/mnist"
	"github.com/fumitoshi0524/ixeoriNet/loss"
	"github.com/fumitoshi0524/ixeoriNet/nn"
	"github.com/fumitoshi0524/ixeoriNet/optim"
)

// optimizer captures the methods needed from any optimizer without importing extra types.
type optimizer interface {
	Step() error
	ZeroGrad()
}

func main() {
	rand.Seed(time.Now().UnixNano())

	trainDS, testDS, err := mnistdata.LoadDefault()
	if err != nil {
		log.Fatalf("load mnist: %v", err)
	}

	classes := 10
	features := trainDS.Features()

	model := nn.NewSequential(
		nn.NewLinear(features, 512, true),
		nn.Relu(),
		nn.NewLinear(512, 256, true),
		nn.Relu(),
		nn.NewLinear(256, classes, true),
	)

	opt := optim.NewAdam(model.Parameters(), 1e-3, 0.9, 0.999, 1e-8)

	if err := trainModel(model, opt, trainDS, testDS, classes); err != nil {
		log.Fatalf("train: %v", err)
	}

	_, testAcc, err := evaluate(model, testDS, 256, classes)
	if err != nil {
		log.Fatalf("evaluate: %v", err)
	}
	if testAcc < 0.97 {
		log.Fatalf("test accuracy %.2f%% below target", testAcc*100)
	}
	fmt.Printf("final test accuracy: %.2f%%\n", testAcc*100)
}

func trainModel(model nn.Module, opt optimizer, trainDS, testDS *mnistdata.Dataset, classes int) error {
	epochs := 12
	batchSize := 128
	samples := trainDS.Count()

	for epoch := 1; epoch <= epochs; epoch++ {
		perm := rand.Perm(samples)
		runningLoss := 0.0
		correct := 0

		for start := 0; start < samples; start += batchSize {
			end := start + batchSize
			if end > samples {
				end = samples
			}
			batchIdx := perm[start:end]
			inputs, targets := trainDS.Batch(batchIdx)

			opt.ZeroGrad()
			preds, err := model.Forward(inputs)
			if err != nil {
				return err
			}
			ce, err := loss.CrossEntropy(preds, targets)
			if err != nil {
				return err
			}
			runningLoss += ce.Data()[0] * float64(len(targets))
			correct += mnistdata.CorrectCount(preds, targets, classes)
			if err := ce.Backward(); err != nil {
				return err
			}
			if err := opt.Step(); err != nil {
				return err
			}
		}

		trainLoss := runningLoss / float64(samples)
		trainAcc := float64(correct) / float64(samples)

		testLoss, testAcc, err := evaluate(model, testDS, 256, classes)
		if err != nil {
			return err
		}

		fmt.Printf("epoch %02d train_loss %.4f train_acc %.2f%% test_loss %.4f test_acc %.2f%%\n",
			epoch, trainLoss, trainAcc*100, testLoss, testAcc*100)
	}

	return nil
}

func evaluate(model nn.Module, ds *mnistdata.Dataset, batchSize, classes int) (float64, float64, error) {
	total := ds.Count()
	totalLoss := 0.0
	totalCorrect := 0

	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		idx := make([]int, end-start)
		for i := start; i < end; i++ {
			idx[i-start] = i
		}
		inputs, targets := ds.Batch(idx)
		preds, err := model.Forward(inputs)
		if err != nil {
			return 0, 0, err
		}
		ce, err := loss.CrossEntropy(preds, targets)
		if err != nil {
			return 0, 0, err
		}
		totalLoss += ce.Data()[0] * float64(len(targets))
		totalCorrect += mnistdata.CorrectCount(preds, targets, classes)
	}

	return totalLoss / float64(total), float64(totalCorrect) / float64(total), nil
}
