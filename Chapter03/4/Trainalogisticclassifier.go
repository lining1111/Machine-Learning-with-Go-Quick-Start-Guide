package main

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/linear"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	mnist "github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/integrate"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"math"
	"math/rand"
)

func main() {
	set, err := mnist.ReadSet("../datasets/mnist/images.gz", "../datasets/mnist/labels.gz")

	//set.Images[1]

	df := MNISTSetToDataframe(set, 1000)

	categories := []string{"tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "shoe", "bag", "boot"}

	training, validation := Split(df, 0.75)

	trainingIsTrouser, err1 := EqualsInt(training.Col("Label"), 1)
	validationIsTrouser, err2 := EqualsInt(validation.Col("Label"), 1)
	if err1 != nil || err2 != nil {
		fmt.Println("Error", err1, err2)
	}

	trainingImages := ImageSeriesToFloats(training, "Image")
	validationImages := ImageSeriesToFloats(validation, "Image")

	model := linear.NewLogistic(base.BatchGA, 1e-4, 1, 150, trainingImages, trainingIsTrouser.Float())

	//Train
	err = model.Learn()
	if err != nil {
		fmt.Println(err)
	}

	//Count correct classifications
	var correct = 0.
	for i := range validationImages {
		prediction, err := model.Predict(validationImages[i])
		if err != nil {
			panic(err)
		}

		if math.Round(prediction[0]) == validationIsTrouser.Elem(i).Float() {
			correct++
		}
	}

	//accuracy
	//correct/float64(len(validationImages))

	//Count true positives and false negatives
	var truePositives = 0.
	var falsePositives = 0.
	var falseNegatives = 0.
	for i := range validationImages {
		prediction, err := model.Predict(validationImages[i])
		if err != nil {
			panic(err)
		}
		if validationIsTrouser.Elem(i).Float() == 1 {
			if math.Round(prediction[0]) == 0 {
				// Predicted false, but actually true
				falseNegatives++
			} else {
				// Predicted true, correctly
				truePositives++
			}
		} else {
			if math.Round(prediction[0]) == 1 {
				// Predicted true, but actually false
				falsePositives++
			}
		}
	}

	//precision
	//truePositives/(truePositives+falsePositives)

	//recall
	//truePositives/(truePositives+falseNegatives)

	model2 := linear.NewSoftmax(base.BatchGA, 1e-4, 1, 10, 100, trainingImages, training.Col("Label").Float())

	//Train
	err = model2.Learn()
	if err != nil {
		fmt.Println(err)
	}
	//create objects for ROC generation
	//as per https://godoc.org/github.com/gonum/stat#ROC
	y := make([][]float64, len(categories), len(categories))
	classes := make([][]bool, len(categories), len(categories))
	//Validate
	for i := 0; i < validation.Col("Image").Len(); i++ {
		prediction, err := model2.Predict(validationImages[i])
		if err != nil {
			panic(err)
		}
		for j := range categories {
			y[j] = append(y[j], prediction[j])
			classes[j] = append(classes[j], validation.Col("Label").Elem(i).Float() != float64(j))
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

func Split(df dataframe.DataFrame, valFraction float64) (training dataframe.DataFrame, validation dataframe.DataFrame) {
	perm := rand.Perm(df.Nrow())
	cutoff := int(valFraction * float64(len(perm)))
	training = df.Subset(perm[:cutoff])
	validation = df.Subset(perm[cutoff:])
	return training, validation
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
	if err := p.Save(5*vg.Inch, 4*vg.Inch, "Multi-class ROC.jpg"); err != nil {
		panic(err)
	}
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	return b.Bytes()
}
