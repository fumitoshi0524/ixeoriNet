package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/fumitoshi0524/ixeoriNet/loss"
	"github.com/fumitoshi0524/ixeoriNet/nn"
	"github.com/fumitoshi0524/ixeoriNet/optim"
	"github.com/fumitoshi0524/ixeoriNet/tensor"
)

const (
	trainImagesFile = "train-images-idx3-ubyte.gz"
	trainLabelsFile = "train-labels-idx1-ubyte.gz"
	testImagesFile  = "t10k-images-idx3-ubyte.gz"
	testLabelsFile  = "t10k-labels-idx1-ubyte.gz"
)

var mnistSources = map[string][]string{
	trainImagesFile: {
		"https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
		"https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
		"https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
	},
	trainLabelsFile: {
		"https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
		"https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
		"https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
	},
	testImagesFile: {
		"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
		"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
		"https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
	},
	testLabelsFile: {
		"https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
		"https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
		"https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
	},
}

type mnistDataset struct {
	images   []float64
	labels   []int
	features int
}

// optimizer captures the methods needed from any optimizer without importing extra types.
type optimizer interface {
	Step() error
	ZeroGrad()
}

func (d *mnistDataset) Count() int {
	return len(d.labels)
}

func (d *mnistDataset) batch(indices []int) (*tensor.Tensor, []int) {
	batchSize := len(indices)
	feat := d.features
	data := make([]float64, batchSize*feat)
	batchLabels := make([]int, batchSize)
	for i, idx := range indices {
		copy(data[i*feat:(i+1)*feat], d.images[idx*feat:(idx+1)*feat])
		batchLabels[i] = d.labels[idx]
	}
	return tensor.MustNew(data, batchSize, feat), batchLabels
}

func main() {
	rand.Seed(time.Now().UnixNano())
	dataDir := filepath.Join("data", "mnist")

	trainDS, testDS, err := loadMNIST(dataDir)
	if err != nil {
		log.Fatalf("load mnist: %v", err)
	}

	classes := 10
	features := trainDS.features

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

func trainModel(model nn.Module, opt optimizer, trainDS, testDS *mnistDataset, classes int) error {
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
			inputs, targets := trainDS.batch(batchIdx)

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
			correct += correctCount(preds, targets, classes)
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

func evaluate(model nn.Module, ds *mnistDataset, batchSize, classes int) (float64, float64, error) {
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
		inputs, targets := ds.batch(idx)
		preds, err := model.Forward(inputs)
		if err != nil {
			return 0, 0, err
		}
		ce, err := loss.CrossEntropy(preds, targets)
		if err != nil {
			return 0, 0, err
		}
		totalLoss += ce.Data()[0] * float64(len(targets))
		totalCorrect += correctCount(preds, targets, classes)
	}

	return totalLoss / float64(total), float64(totalCorrect) / float64(total), nil
}

func correctCount(logits *tensor.Tensor, labels []int, classes int) int {
	data := logits.Data()
	batch := len(labels)
	correct := 0
	for i := 0; i < batch; i++ {
		base := i * classes
		bestIdx := 0
		bestVal := data[base]
		for j := 1; j < classes; j++ {
			v := data[base+j]
			if v > bestVal {
				bestVal = v
				bestIdx = j
			}
		}
		if bestIdx == labels[i] {
			correct++
		}
	}
	return correct
}

func loadMNIST(dir string) (*mnistDataset, *mnistDataset, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, nil, err
	}

	trainImgsPath, err := downloadIfMissing(dir, trainImagesFile)
	if err != nil {
		return nil, nil, err
	}
	trainLblsPath, err := downloadIfMissing(dir, trainLabelsFile)
	if err != nil {
		return nil, nil, err
	}
	testImgsPath, err := downloadIfMissing(dir, testImagesFile)
	if err != nil {
		return nil, nil, err
	}
	testLblsPath, err := downloadIfMissing(dir, testLabelsFile)
	if err != nil {
		return nil, nil, err
	}

	trainImages, trainCount, features, err := loadImages(trainImgsPath)
	if err != nil {
		return nil, nil, err
	}
	trainLabels, err := loadLabels(trainLblsPath)
	if err != nil {
		return nil, nil, err
	}
	if len(trainLabels) != trainCount {
		return nil, nil, fmt.Errorf("train label count mismatch: got %d want %d", len(trainLabels), trainCount)
	}

	testImages, testCount, testFeatures, err := loadImages(testImgsPath)
	if err != nil {
		return nil, nil, err
	}
	testLabels, err := loadLabels(testLblsPath)
	if err != nil {
		return nil, nil, err
	}
	if len(testLabels) != testCount {
		return nil, nil, fmt.Errorf("test label count mismatch: got %d want %d", len(testLabels), testCount)
	}
	if features != testFeatures {
		return nil, nil, fmt.Errorf("feature mismatch: train %d test %d", features, testFeatures)
	}

	trainDS := &mnistDataset{images: trainImages, labels: trainLabels, features: features}
	testDS := &mnistDataset{images: testImages, labels: testLabels, features: features}
	return trainDS, testDS, nil
}

func downloadIfMissing(dir, filename string) (string, error) {
	path := filepath.Join(dir, filename)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}

	urls, ok := mnistSources[filename]
	if !ok || len(urls) == 0 {
		return "", fmt.Errorf("no download sources for %s", filename)
	}

	var lastErr error
	for _, url := range urls {
		fmt.Printf("downloading %s...\n", url)
		if err := downloadFile(url, path); err != nil {
			fmt.Printf("failed %s: %v\n", url, err)
			lastErr = err
			continue
		}
		return path, nil
	}

	return "", fmt.Errorf("failed to download %s: %w", filename, lastErr)
}

func downloadFile(url, path string) error {
	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %s", resp.Status)
	}

	tmpPath := path + ".tmp"
	tmpFile, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return err
	}
	if err := tmpFile.Close(); err != nil {
		os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return err
	}
	return nil
}

func loadImages(path string) ([]float64, int, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, 0, 0, err
	}
	defer gz.Close()

	var magic uint32
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return nil, 0, 0, err
	}
	if magic != 2051 {
		return nil, 0, 0, fmt.Errorf("unexpected magic number %d for images", magic)
	}

	var num, rows, cols uint32
	if err := binary.Read(gz, binary.BigEndian, &num); err != nil {
		return nil, 0, 0, err
	}
	if err := binary.Read(gz, binary.BigEndian, &rows); err != nil {
		return nil, 0, 0, err
	}
	if err := binary.Read(gz, binary.BigEndian, &cols); err != nil {
		return nil, 0, 0, err
	}

	total := int(num) * int(rows) * int(cols)
	raw := make([]byte, total)
	if _, err := io.ReadFull(gz, raw); err != nil {
		return nil, 0, 0, err
	}

	data := make([]float64, total)
	for i, b := range raw {
		data[i] = float64(b) / 255.0
	}

	return data, int(num), int(rows) * int(cols), nil
}

func loadLabels(path string) ([]int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, err
	}
	defer gz.Close()

	var magic uint32
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != 2049 {
		return nil, fmt.Errorf("unexpected magic number %d for labels", magic)
	}

	var num uint32
	if err := binary.Read(gz, binary.BigEndian, &num); err != nil {
		return nil, err
	}

	raw := make([]byte, num)
	if _, err := io.ReadFull(gz, raw); err != nil {
		return nil, err
	}

	labels := make([]int, num)
	for i, b := range raw {
		labels[i] = int(b)
	}
	return labels, nil
}
