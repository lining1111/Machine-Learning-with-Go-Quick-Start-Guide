package main

import (
	"encoding/json"
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	mnist "github.com/petar/GoMNIST"
	"io/ioutil"
	"os/exec"
)

func main() {
	set, err := mnist.ReadSet("../datasets/mnist/images.gz", "../datasets/mnist/labels.gz")
	if err != nil {
		panic(err)
	}

	//set.Images[1]

	df := MNISTSetToDataframe(set, 1000)

	//categories := []string{"tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "shoe", "bag", "boot"}

	testImages := ImageSeriesToInts(df, "Image")

	//  Prediction
	IsImageTrousers(testImages[16])

	// Ground truth
	//df.Col("Label").Elem(16).Int()==1

	//  Prediction
	IsImageTrousers(testImages[0])

	// Ground truth
	//df.Col("Label").Elem(0).Int()==1

}

func MNISTSetToDataframe(st *mnist.Set, maxExamples int) dataframe.DataFrame {
	length := maxExamples
	if length > len(st.Images) {
		length = len(st.Images)
	}
	s := make([]string, length, length)
	l := make([]int, length, length)
	for i := 0; i < length; i++ {
		s[i] = string(st.Images[i])
		l[i] = int(st.Labels[i])
	}
	var df dataframe.DataFrame
	images := series.Strings(s)
	images.Name = "Image"
	labels := series.Ints(l)
	labels.Name = "Label"
	df = dataframe.New(images, labels)
	return df
}

func NormalizeBytes(bs []byte) []int {
	ret := make([]int, len(bs), len(bs))
	for i := range bs {
		ret[i] = int(bs[i])
	}
	return ret
}
func ImageSeriesToInts(df dataframe.DataFrame, col string) [][]int {
	s := df.Col(col)
	ret := make([][]int, s.Len(), s.Len())
	for i := 0; i < s.Len(); i++ {
		b := []byte(s.Elem(i).String())
		ret[i] = NormalizeBytes(b)
	}
	return ret
}

func InvokeAndWait(args ...string) ([]byte, error) {
	var (
		output    []byte
		errOutput []byte
		err       error
	)
	cmd := exec.Command("python3", args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	if output, err = ioutil.ReadAll(stdout); err != nil {
		return nil, err
	}

	if errOutput, err = ioutil.ReadAll(stderr); err != nil || len(errOutput) > 0 {
		return nil, fmt.Errorf("Error running model: %s", string(errOutput))
	}

	return output, nil
}

// IsImageTrousers invokes the Python model to predict if image at given index is, in fact, of trousers
func IsImageTrousers(testImage []int) (bool, error) {
	b, err := json.Marshal(testImage)
	if err != nil {
		panic(err)
	}
	b, err = InvokeAndWait("model.py", "predict", string(b))
	if err != nil {
		return false, err
	} else {
		var ret struct {
			IsTrousers bool `json:"is_trousers"`
		}
		err := json.Unmarshal(b, &ret)
		if err != nil {
			return false, err
		}
		return ret.IsTrousers, nil
	}
}
