package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

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

// DefaultDir returns an OS-specific cache directory for storing MNIST assets.
func DefaultDir() string {
	return filepath.Join(os.TempDir(), "ixeoriNet", "mnist")
}

// LoadDefault downloads (if needed) and loads MNIST from the default cache path.
func LoadDefault() (*Dataset, *Dataset, error) {
	return Load(DefaultDir())
}

// Dataset stores normalized MNIST samples and labels.
type Dataset struct {
	images   []float64
	labels   []int
	features int
	height   int
	width    int
}

// Load reads the MNIST dataset from disk, downloading it if missing.
func Load(dir string) (*Dataset, *Dataset, error) {
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

	trainImages, trainCount, rows, cols, err := loadImages(trainImgsPath)
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

	testImages, testCount, testRows, testCols, err := loadImages(testImgsPath)
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
	if rows != testRows || cols != testCols {
		return nil, nil, fmt.Errorf("feature mismatch: train %dx%d test %dx%d", rows, cols, testRows, testCols)
	}

	features := rows * cols
	trainDS := &Dataset{images: trainImages, labels: trainLabels, features: features, height: rows, width: cols}
	testDS := &Dataset{images: testImages, labels: testLabels, features: features, height: rows, width: cols}
	return trainDS, testDS, nil
}

// Count returns the number of samples in the dataset.
func (d *Dataset) Count() int {
	if d == nil {
		return 0
	}
	return len(d.labels)
}

// Features returns the flattened feature dimension per sample.
func (d *Dataset) Features() int {
	return d.features
}

// Height returns the image height.
func (d *Dataset) Height() int {
	return d.height
}

// Width returns the image width.
func (d *Dataset) Width() int {
	return d.width
}

// Batch materializes a batch of samples into a tensor shaped [batch, features].
func (d *Dataset) Batch(indices []int) (*tensor.Tensor, []int) {
	batchSize := len(indices)
	feat := d.features
	data := make([]float64, batchSize*feat)
	batchLabels := make([]int, batchSize)
	for i, idx := range indices {
		start := idx * feat
		end := start + feat
		copy(data[i*feat:(i+1)*feat], d.images[start:end])
		batchLabels[i] = d.labels[idx]
	}
	return tensor.MustNew(data, batchSize, feat), batchLabels
}

// CorrectCount returns the number of correct predictions for the provided logits and labels.
func CorrectCount(logits *tensor.Tensor, labels []int, classes int) int {
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

func loadImages(path string) ([]float64, int, int, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer file.Close()

	gz, err := gzip.NewReader(file)
	if err != nil {
		return nil, 0, 0, 0, err
	}
	defer gz.Close()

	var magic uint32
	if err := binary.Read(gz, binary.BigEndian, &magic); err != nil {
		return nil, 0, 0, 0, err
	}
	if magic != 2051 {
		return nil, 0, 0, 0, fmt.Errorf("unexpected magic number %d for images", magic)
	}

	var num, rows, cols uint32
	if err := binary.Read(gz, binary.BigEndian, &num); err != nil {
		return nil, 0, 0, 0, err
	}
	if err := binary.Read(gz, binary.BigEndian, &rows); err != nil {
		return nil, 0, 0, 0, err
	}
	if err := binary.Read(gz, binary.BigEndian, &cols); err != nil {
		return nil, 0, 0, 0, err
	}

	total := int(num) * int(rows) * int(cols)
	raw := make([]byte, total)
	if _, err := io.ReadFull(gz, raw); err != nil {
		return nil, 0, 0, 0, err
	}

	data := make([]float64, total)
	for i, b := range raw {
		data[i] = float64(b) / 255.0
	}

	return data, int(num), int(rows), int(cols), nil
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
