package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	mnist "github.com/petar/GoMNIST"
	"math/rand"
)

var (
	trainingExamples   []training.Example
	validationExamples []training.Example
)

func main() {
	set, err := mnist.ReadSet("../datasets/mnist/images.gz", "../datasets/mnist/labels.gz")
	if err != nil {
		panic(err)
	}
	//set.Images[1]

	df := MNISTSetToDataframe(set, 1000)

	categories := []string{"tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "shoe", "bag", "boot"}

	train, validation := Split(df, 0.75)

	trainingImages := ImageSeriesToFloats(train, "Image")
	validationImages := ImageSeriesToFloats(validation, "Image")

	trainingOutputs := make([][]float64, len(trainingImages))
	validationOutputs := make([][]float64, len(validationImages))

	ltCol := train.Col("Label")
	for i := range trainingImages {
		l := make([]float64, len(categories), len(categories))
		val, _ := ltCol.Elem(i).Int()
		l[val] = 1
		trainingOutputs[i] = l
	}

	lvCol := validation.Col("Label")
	for i := range validationImages {
		l := make([]float64, len(categories), len(categories))
		val, _ := lvCol.Elem(i).Int()
		l[val] = 1
		validationOutputs[i] = l
	}

	for i := range trainingImages {
		trainingExamples = append(trainingExamples, training.Example{trainingImages[i], trainingOutputs[i]})
	}
	for i := range validationImages {
		validationExamples = append(validationExamples, training.Example{validationImages[i], validationOutputs[i]})
	}

	network := deep.NewNeural(&deep.Config{
		// Input size: 784 in our case (number of pixels in each image)
		Inputs: len(trainingImages[0]),
		// Two hidden layers of 128 neurons each, and an output layer 10 neurons (one for each class)
		Layout: []int{128, 128, len(categories)},
		// ReLU activation to introduce some additional non-linearity
		Activation: deep.ActivationReLU,
		// We need a multi-class model
		Mode: deep.ModeMultiClass,
		// Initialise the weights of each neuron using normally distributed random numbers
		Weight: deep.NewNormal(0.5, 0.1),
		Bias:   true,
	})

	// Parameters: learning rate, momentum, alpha decay, nesterov
	optimizer := training.NewSGD(0.006, 0.1, 1e-6, true)
	trainer := training.NewTrainer(optimizer, 1)

	trainer.Train(network, trainingExamples, validationExamples, 500) // training, validation, iterations

	validCorrect := 0.
	for i := range validationImages {
		prediction := network.Predict(validationImages[i])

		if MaxIndex(prediction) == MaxIndex(validationOutputs[i]) {
			validCorrect++
		}
	}
	fmt.Printf("Validation Accuracy: %5.2f\n", validCorrect/float64(len(validationImages)))
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

func EqualsInt(s series.Series, to int) (*series.Series, error) {
	eq := make([]int, s.Len(), s.Len())
	ints, err := s.Int()
	if err != nil {
		return nil, err
	}
	for i := range ints {
		if ints[i] == to {
			eq[i] = 1
		}
	}
	ret := series.Ints(eq)
	return &ret, nil
}

func Split(df dataframe.DataFrame, valFraction float64) (training dataframe.DataFrame, validation dataframe.DataFrame) {
	perm := rand.Perm(df.Nrow())
	cutoff := int(valFraction * float64(len(perm)))
	training = df.Subset(perm[:cutoff])
	validation = df.Subset(perm[cutoff:])
	return training, validation
}

func NormalizeBytes(bs []byte) []float64 {
	ret := make([]float64, len(bs), len(bs))
	for i := range bs {
		ret[i] = float64(bs[i]) / 255.
	}
	return ret
}

func ImageSeriesToFloats(df dataframe.DataFrame, col string) [][]float64 {
	s := df.Col(col)
	ret := make([][]float64, s.Len(), s.Len())
	for i := 0; i < s.Len(); i++ {
		b := []byte(s.Elem(i).String())
		ret[i] = NormalizeBytes(b)
	}
	return ret
}

func MaxIndex(f []float64) (i int) {
	var (
		curr float64
		ix   int = -1
	)
	for i := range f {
		if f[i] > curr {
			curr = f[i]
			ix = i
		}
	}
	return ix
}
