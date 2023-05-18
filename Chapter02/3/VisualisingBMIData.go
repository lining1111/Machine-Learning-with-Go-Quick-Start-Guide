package main

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"io/ioutil"
)

const path = "../datasets/bmi/500_Person_Gender_Height_Weight_Index.csv"

func main() {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println("Error!", err)
	}
	df := dataframe.ReadCSV(bytes.NewReader(b))

	fmt.Println("Minimum", df.Col("Height").Min())
	fmt.Println("Maximum", df.Col("Height").Max())
	fmt.Println("Mean", df.Col("Height").Mean())
	fmt.Println("Median", df.Col("Height").Quantile(0.5))

	df.Describe()

	//display.JPEG(histogramBytes(SeriesToPlotValues(df, "Height"), "Height Histogram"))
	//
	//display.JPEG(histogramBytes(SeriesToPlotValues(df, "Weight"), "Weight Histogram"))
	//
	//display.JPEG(histogramBytes(SeriesToPlotValues(df, "Index"), "BMI Histogram"))
}

// SeriesToPlotValues takes a column of a Dataframe and converts it to a gonum/plot/plotter.Values slice.
// Panics if the column does not exist.
func SeriesToPlotValues(df dataframe.DataFrame, col string) plotter.Values {
	rows, _ := df.Dims()
	v := make(plotter.Values, rows)
	s := df.Col(col)
	for i := 0; i < rows; i++ {
		v[i] = s.Elem(i).Float()
	}
	return v
}

//  showHistogram plots a histogram of the column with name col in the dataframe df.
func histogramBytes(v plotter.Values, title string) []byte {
	// Make a plot and set its title.
	p := plot.New()

	p.Title.Text = title
	h, err := plotter.NewHist(v, 10)
	if err != nil {
		panic(err)
	}
	//h.Normalize(1)
	p.Add(h)
	w, err := p.WriterTo(5*vg.Inch, 4*vg.Inch, "jpg")
	if err != nil {
		panic(err)
	}
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	if err := p.Save(5*vg.Inch, 4*vg.Inch, title+".ong"); err != nil {
		panic(err)
	}
	return b.Bytes()
}
