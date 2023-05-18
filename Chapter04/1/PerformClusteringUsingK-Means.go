package main

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/cdipaolo/goml/base"
	"github.com/cdipaolo/goml/cluster"
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
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

	features, classification := DataFrameToXYs(df, "species")

	model := cluster.NewKMeans(3, 30, features)

	if err := model.Learn(); err != nil {
		panic(err)
	}

	scatterData, labels := PredictionsToScatterData(features, classification, model, 2, 3)

	b1, _ := PlotClusterData(scatterData, labels, "Sepal length", "Sepal width")
	fmt.Println(b1)
	//display.JPEG(b)
}

//  DataFrameToXYs converts a dataframe with float64 columns to a slice of independent variable columns as floats
//  and the dependent variable (yCol). This can then be used with eg. goml's linear ML algorithms.
//  yCol is optional - if it doesn't exist only the x (independent) variables will be returned.
func DataFrameToXYs(df dataframe.DataFrame, yCol string) ([][]float64, []float64) {
	var (
		x      [][]float64
		y      []float64
		yColIx = -1
	)

	//find dependent variable column index
	for i, col := range df.Names() {
		if col == yCol {
			yColIx = i
			break
		}
	}
	if yColIx == -1 {
		fmt.Println("Warning - no dependent variable")
	}
	x = make([][]float64, df.Nrow(), df.Nrow())
	y = make([]float64, df.Nrow())
	for i := 0; i < df.Nrow(); i++ {
		var xx []float64
		for j := 0; j < df.Ncol(); j++ {
			if j == yColIx {
				y[i] = df.Elem(i, j).Float()
				continue
			}
			xx = append(xx, df.Elem(i, j).Float())
		}
		x[i] = xx
	}
	return x, y
}

//  PredictionsToScatterData gets predictions from the model based on the features and converts to map from label to XYs
func PredictionsToScatterData(features [][]float64, labels []float64, model base.Model, featureForXAxis, featureForYAxis int) (map[int]plotter.XYs, map[int][]float64) {
	ret := make(map[int]plotter.XYs)
	labelMap := make(map[int][]float64)
	if features == nil {
		panic("No features to plot")
	}

	for i := range features {
		var pt struct{ X, Y float64 }
		pt.X = features[i][featureForXAxis]
		pt.Y = features[i][featureForYAxis]
		p, _ := model.Predict(features[i])
		labelMap[int(p[0])] = append(labelMap[int(p[0])], labels[i])
		ret[int(p[0])] = append(ret[int(p[0])], pt)
	}
	return ret, labelMap
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

func PlotClusterData(labelsToXYs map[int]plotter.XYs, classes map[int][]float64, xLabel, yLabel string) ([]uint8, error) {
	p := plot.New()

	p.Title.Text = "Iris Dataset K-Means Example"
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
		s.GlyphStyleFunc = func(ii int) func(jj int) draw.GlyphStyle {
			return func(j int) draw.GlyphStyle {
				var gs draw.GlyphStyle
				if j >= len(classes[ii]) {
					gs.Shape = plotutil.Shape(10)
				} else {
					gs.Shape = plotutil.Shape(int(classes[ii][j]))
				}
				gs.Color = plotutil.Color(ii)
				gs.Radius = 2.
				return gs
			}
		}(i)
		//s.Color = plotutil.Color(i)
		//s.Shape = plotutil.Shape(i)
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
	if err := p.Save(5*vg.Inch, 4*vg.Inch, "Clustering Scatter.jpg"); err != nil {
		return nil, err
	}
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	return b.Bytes(), nil
}
