package main

import (
	"bytes"
	"fmt"
	"github.com/fxsjy/RF.go/RF/Regression"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"gonum.org/v1/gonum/stat"
	"io/ioutil"
	"math/rand"
)

const path = "../datasets/housing/CaliforniaHousing/cal_housing.data"

func main() {
	columns := []string{"longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms", "population", "households", "medianIncome", "medianHouseValue"}
	b, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println("Error!", err)
	}
	df := dataframe.ReadCSV(bytes.NewReader(b), dataframe.Names(columns...))
	df = df.Mutate(Divide(df.Col("totalRooms"), df.Col("households"), "averageRooms"))
	df = df.Mutate(Divide(df.Col("totalBedrooms"), df.Col("households"), "averageBedrooms"))
	df = df.Mutate(Divide(df.Col("population"), df.Col("households"), "averageOccupancy"))
	df = df.Mutate(MultiplyConst(df.Col("medianHouseValue"), 0.00001))
	df = df.Select([]string{"medianIncome", "housingMedianAge", "averageRooms", "averageBedrooms", "population", "averageOccupancy", "latitude", "longitude", "medianHouseValue"})

	training, validation := Split(df, 0.75)

	tx, trainingY := DataFrameToXYs(training, "medianHouseValue")
	vx, validationY := DataFrameToXYs(validation, "medianHouseValue")

	var (
		trainingX   = make([][]interface{}, len(tx), len(tx))
		validationX = make([][]interface{}, len(vx), len(vx))
	)

	for i := range tx {
		trainingX[i] = FloatsToInterfaces(tx[i])
	}
	for i := range vx {
		validationX[i] = FloatsToInterfaces(vx[i])
	}

	model := Regression.BuildForest(trainingX, trainingY, 25, len(trainingX), 1)

	//On validation set
	errors := make([]float64, len(validationX), len(validationX))
	for i := range validationX {
		prediction := model.Predicate(validationX[i])
		if err != nil {
			panic("Prediction error " + err.Error())
		}
		errors[i] = (prediction - validationY[i]) * (prediction - validationY[i])
	}

	fmt.Printf("MSE: %5.2f\n", stat.Mean(errors, nil))

	// On training set
	errors = make([]float64, len(trainingX), len(trainingX))
	for i := range trainingX {
		prediction := model.Predicate(trainingX[i])
		if err != nil {
			panic("Prediction error " + err.Error())
		}
		errors[i] = (prediction - trainingY[i]) * (prediction - trainingY[i])
	}

	fmt.Printf("MSE: %5.2f\n", stat.Mean(errors, nil))

}

// Divide divides two series and returns a series with the given name. The series must have the same length.
func Divide(s1 series.Series, s2 series.Series, name string) series.Series {
	if s1.Len() != s2.Len() {
		panic("Series must have the same length!")
	}

	ret := make([]interface{}, s1.Len(), s1.Len())
	for i := 0; i < s1.Len(); i++ {
		ret[i] = s1.Elem(i).Float() / s2.Elem(i).Float()
	}
	s := series.Floats(ret)
	s.Name = name
	return s
}

//  MultiplyConst multiplies the series by a constant and returns another series with the same name.
func MultiplyConst(s series.Series, f float64) series.Series {
	ret := make([]interface{}, s.Len(), s.Len())
	for i := 0; i < s.Len(); i++ {
		ret[i] = s.Elem(i).Float() * f
	}
	ss := series.Floats(ret)
	ss.Name = s.Name
	return ss
}

func Split(df dataframe.DataFrame, valFraction float64) (training dataframe.DataFrame, validation dataframe.DataFrame) {
	perm := rand.Perm(df.Nrow())
	cutoff := int(valFraction * float64(len(perm)))
	training = df.Subset(perm[:cutoff])
	validation = df.Subset(perm[cutoff:])
	return training, validation
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

func FloatsToInterfaces(f []float64) []interface{} {
	iif := make([]interface{}, len(f), len(f))
	for i := range f {
		iif[i] = f[i]
	}
	return iif
}
