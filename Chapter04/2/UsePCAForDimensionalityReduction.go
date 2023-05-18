package main

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"io/ioutil"
	"strconv"
)

const path = "../datasets/iris/iris.csv"

func main() {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println("Error!", err)
	}
	df := dataframe.ReadCSV(bytes.NewReader(b))
	df.SetNames("petal length", "petal width", "sepal length", "sepal width", "species")

	df = Standardise(df, "petal length")
	df = Standardise(df, "petal width")
	df = Standardise(df, "sepal length")
	df = Standardise(df, "sepal width")
	labels := df.Col("species").Float()
	df = DropColumn(df, "species")
	features := DataFrameToMatrix(df)

	model := stat.PC{}
	if ok := model.PrincipalComponents(features, nil); !ok {
		fmt.Println("Error!")
	}

	var variances []float64
	model.VarsTo(variances)
	components := &mat.Dense{}
	model.VectorsTo(components)

	// Print the amount of variance explained by each component
	total_variance := 0.0
	for i := range variances {
		total_variance += variances[i]
	}

	for i := range variances {
		fmt.Printf("Component %d: %5.3f\n", i+1, variances[i]/total_variance)
	}

	// Transform the data into the new space
	transform := mat.NewDense(df.Nrow(), 4, nil)
	transform.Mul(features, components)

	scatterData := PCAToScatterData(transform, labels)

	b1, _ := PlotPCAData(scatterData, "Component 1", "Component 2")
	fmt.Println(b1)
	//display.JPEG(b)
}

//  Standardise maps the given column values by subtracting mean and rescaling by standard deviation
func Standardise(df dataframe.DataFrame, col string) dataframe.DataFrame {
	s := df.Col(col)
	std := s.StdDev()
	mean := s.Mean()
	v := make([]float64, s.Len(), s.Len())
	for i := 0; i < s.Len(); i++ {
		v[i] = (s.Elem(i).Float() - mean) / std
	}
	rs := series.Floats(v)
	rs.Name = col
	return df.Mutate(rs)
}

//  DropColumn returns a new dataframe that does not include the given column
func DropColumn(df dataframe.DataFrame, col string) dataframe.DataFrame {
	var s []series.Series
	for _, c := range df.Names() {
		if c == col {
			continue
		}
		s = append(s, df.Col(c))
	}
	return dataframe.New(s...)
}

//  DataFrameToMatrix converts the given dataframe to a gonum matrix
func DataFrameToMatrix(df dataframe.DataFrame) mat.Matrix {
	var x []float64 //slice to hold matrix entries in row-major order

	for i := 0; i < df.Nrow(); i++ {
		for j := 0; j < df.Ncol(); j++ {
			x = append(x, df.Elem(i, j).Float())
		}
	}
	return mat.NewDense(df.Nrow(), df.Ncol(), x)
}

//  PCA keeps top 2 components (one for x axis, one for y) and returns this as map from label to XYs
//  Matrix must have at least 2 columns or this will panic
func PCAToScatterData(m mat.Matrix, labels []float64) map[int]plotter.XYs {
	ret := make(map[int]plotter.XYs)
	nrows, _ := m.Dims()
	for i := 0; i < nrows; i++ {
		var pt struct{ X, Y float64 }
		pt.X = m.At(i, 0)
		pt.Y = m.At(i, 1)
		ret[int(labels[i])] = append(ret[int(labels[i])], pt)
	}
	return ret
}

/**
  NB. This is required because gophernotes comes with an old version of goml. When it gets updated we can remove most of this.
*/

type LegacyXYs plotter.XYs

func (xys LegacyXYs) Len() int {
	return len(xys)
}

func (xys LegacyXYs) XY(i int) (float64, float64) {
	return xys[i].X, xys[i].Y
}

func PlotPCAData(labelsToXYs map[int]plotter.XYs, xLabel, yLabel string) ([]uint8, error) {
	p := plot.New()

	p.Title.Text = "Iris Dataset PCA Example"
	//p.X.Min = 4
	//p.X.Max = 9
	p.X.Padding = 0
	p.X.Label.Text = xLabel
	//p.Y.Min = 1.5
	//p.Y.Max = 4.5
	p.Y.Padding = 0
	p.Y.Label.Text = yLabel
	for i := range labelsToXYs {
		s, err := plotter.NewScatter(LegacyXYs(labelsToXYs[i])) //Remove LegacyXYs when gophernotes updated to use latest goml
		s.Color = plotutil.Color(i)
		s.Shape = plotutil.Shape(i)
		p.Add(s)
		n := strconv.Itoa(i)
		p.Legend.Add(n)
		if err != nil {
			return nil, err
		}
	}
	w, err := p.WriterTo(5*vg.Inch, 4*vg.Inch, "jpg")
	if err != nil {
		return nil, err
	}
	if err := p.Save(5*vg.Inch, 4*vg.Inch, "PCA Scatter.jpg"); err != nil {
		return nil, err
	}
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	return b.Bytes(), nil
}
