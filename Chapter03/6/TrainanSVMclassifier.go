package main

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/datastream/libsvm"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	mnist "github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/integrate"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"math/rand"
	"os"
)

func main() {
	if os.Getenv("GODEBUG") != "cgocheck=0" {
		fmt.Println("WARNING: GODEBUG Not set to cgocheck=0. This example will probably not work!")
	}

	set, err := mnist.ReadSet("../datasets/mnist/images.gz", "../datasets/mnist/labels.gz")
	if err != nil {
		panic(err)
	}

	//set.Images[1]

	df := MNISTSetToDataframe(set, 1000)

	categories := []string{"tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "shoe", "bag", "boot"}

	training, validation := Split(df, 0.75)

	_, err1 := EqualsInt(training.Col("Label"), 1)
	_, err2 := EqualsInt(validation.Col("Label"), 1)
	if err1 != nil || err2 != nil {
		fmt.Println("Error", err1, err2)
	}

	//construct outputs
	trainingImages := ImageSeriesToFloats(training, "Image")
	validationImages := ImageSeriesToFloats(validation, "Image")

	trainingOutputs := make([]float64, len(trainingImages))
	validationOutputs := make([]float64, len(validationImages))

	ltCol := training.Col("Label")
	for i := range trainingImages {
		trainingOutputs[i] = ltCol.Elem(i).Float()
	}

	lvCol := validation.Col("Label")
	for i := range validationImages {
		validationOutputs[i] = lvCol.Elem(i).Float()
	}

	//construct inputs following https://github.com/cjlin1/libsvm
	var (
		trainingProblem   libsvm.SVMProblem
		validationProblem libsvm.SVMProblem
	)

	trainingProblem.L = len(trainingImages)
	validationProblem.L = len(validationImages)
	for i := range trainingImages {
		trainingProblem.X = append(trainingProblem.X, FloatsToSVMNode(trainingImages[i]))
	}
	trainingProblem.Y = trainingOutputs

	for i := range validationImages {
		validationProblem.X = append(validationProblem.X, FloatsToSVMNode(validationImages[i]))
	}
	validationProblem.Y = validationOutputs

	//  configure SVM
	svm := libsvm.NewSvm()
	//  From Python notebook
	param := libsvm.SVMParameter{
		SvmType:     libsvm.CSVC,
		KernelType:  libsvm.RBF,
		C:           100,
		Gamma:       0.01,
		Coef0:       0,
		Degree:      3,
		Eps:         0.001,
		Probability: 1,
	}

	model := svm.SVMTrain(&trainingProblem, &param)

	var (
		trainCorrect float64
		validCorrect float64
	)
	predictions := make([]int, len(validationProblem.X), len(validationProblem.X))
	p := make([]float64, len(categories), len(categories)) // pre-allocated probability slice
	// for ROC
	probs := make([][]float64, len(validationProblem.X), len(validationProblem.X))

	for i := range trainingProblem.X {
		prediction := svm.SVMPredictProbability(model, trainingProblem.X[i], p)
		if prediction == trainingProblem.Y[i] {
			trainCorrect++
		}
	}

	for i := range validationProblem.X {
		prediction := svm.SVMPredictProbability(model, validationProblem.X[i], p)
		probs[i] = make([]float64, len(categories))
		copy(probs[i], p)
		predictions[i] = int(prediction)
		if prediction == validationProblem.Y[i] {
			validCorrect++
		}
	}
	fmt.Printf("Train Accuracy: %5.2f\n", trainCorrect/float64(len(trainingProblem.X)))
	fmt.Printf("Validation Accuracy: %5.2f\n", validCorrect/float64(len(validationProblem.X)))

	//create objects for ROC generation
	//as per https://godoc.org/github.com/gonum/stat#ROC
	y := make([][]float64, len(categories), len(categories))
	classes := make([][]bool, len(categories), len(categories))
	labels := model.Label
	for i := range y {
		classes[i] = make([]bool, len(probs), len(probs))
	}

	for i := range probs {
		for j := range categories {
			y[labels[j]] = append(y[labels[j]], probs[i][j])
			classes[labels[j]][i] = float64(labels[j]) != validationProblem.Y[i]
		}

	}

	//Calculate ROC
	tprs := make([][]float64, len(categories), len(categories))
	fprs := make([][]float64, len(categories), len(categories))

	for i := range categories {
		stat.SortWeightedLabeled(y[i], classes[i], nil)
		cutoffs := make([]float64, len(categories))
		tprs[i], fprs[i], _ = stat.ROC(cutoffs, y[i], classes[i], nil)
	}

	for i := range categories {
		fmt.Println(categories[i])
		auc := integrate.Trapezoidal(fprs[i], tprs[i])
		fmt.Println(auc)
	}

	//display.JPEG(plotROCBytes(fprs, tprs, categories))

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

//  FloatstoSVMNode converts a slice of float64 to SVMNode with sequential indices starting at 1
func FloatsToSVMNode(f []float64) []libsvm.SVMNode {
	ret := make([]libsvm.SVMNode, len(f), len(f))
	for i := range f {
		ret[i] = libsvm.SVMNode{
			Index: i + 1,
			Value: f[i],
		}
	}
	//End of Vector
	ret = append(ret, libsvm.SVMNode{
		Index: -1,
		Value: 0,
	})
	return ret
}

func plotROCBytes(fprs, tprs [][]float64, labels []string) []byte {
	p := plot.New()

	p.Title.Text = "ROC Curves"
	p.X.Label.Text = "False Positive Rate"
	p.Y.Label.Text = "True Positive Rate"

	for i := range labels {
		pts := make(plotter.XYs, len(fprs[i]))
		for j := range fprs[i] {
			pts[j].X = fprs[i][j]
			pts[j].Y = tprs[i][j]
		}
		lines, points, err := plotter.NewLinePoints(pts)
		if err != nil {
			panic(err)
		}
		lines.Color = plotutil.Color(i)
		lines.Width = 2
		points.Shape = nil

		p.Add(lines, points)
		p.Legend.Add(labels[i], lines, points)
	}

	w, err := p.WriterTo(5*vg.Inch, 4*vg.Inch, "jpg")
	if err != nil {
		panic(err)
	}
	if err := p.Save(5*vg.Inch, 4*vg.Inch, "SVM ROC.jpg"); err != nil {
		panic(err)
	}
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	return b.Bytes()
}
