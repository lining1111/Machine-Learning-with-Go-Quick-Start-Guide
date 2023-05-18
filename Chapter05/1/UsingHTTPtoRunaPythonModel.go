package main

import (
	"bytes"
	"encoding/json"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	mnist "github.com/petar/GoMNIST"
	"io/ioutil"
	"net/http"
	"net/rpc/jsonrpc"
)

type PredictRequest struct {
	Image []int
}

func main() {
	set, err := mnist.ReadSet("../datasets/mnist/images.gz", "../datasets/mnist/labels.gz")
	if err != nil {
		panic(err)
	}

	//set.Images[1]

	df := MNISTSetToDataframe(set, 1000)

	//categories := []string{"tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "shoe", "bag", "boot"}

	testImages := ImageSeriesToInts(df, "Image")

	c, err := jsonrpc.Dial("tcp", "localhost:8001")
	//p := model{Client: client}
	var req PredictRequest = PredictRequest{
		Image: testImages[16],
	}

	var reply interface{}
	c.Call("predict", req, &reply)
	// Expected: true <nil>
	Predict(testImages[16])

	// Expected false <nil>
	Predict(testImages[0])

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

//  Predict returns whether the ith image represents trousers or not based on the logistic regression model
func Predict(testImage []int) (bool, error) {
	b, err := json.Marshal(testImage)
	if err != nil {
		return false, err
	}
	r := bytes.NewReader(b)
	resp, err := http.Post("http://127.0.0.1:8001", "application/json", r)
	if err != nil {
		return false, err
	}
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return false, err
	}
	resp.Body.Close()
	var resp1 struct {
		IsTrousers bool `json:"is_trousers"`
	}
	err = json.Unmarshal(body, &resp1)
	return resp1.IsTrousers, err
}
