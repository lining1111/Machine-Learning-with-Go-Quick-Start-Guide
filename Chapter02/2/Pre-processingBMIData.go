package main

import (
	"bytes"
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"io/ioutil"
	"math/rand"
)

const path = "../datasets/bmi/SOCR_Data_MLB_HeightsWeights.csv"

func main() {
	b, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println("Error!", err)
	}
	df := dataframe.ReadCSV(bytes.NewReader(b))

	df = df.Select([]string{"Position", "Height(inches)", "Weight(pounds)", "Age"})
	df = df.Rename("Height", "Height(inches)")
	df = df.Rename("Weight", "Weight(pounds)")

	df = df.Mutate(series.New(df.Col("Height"), series.Float, "Height"))
	df = df.Mutate(series.New(df.Col("Weight"), series.Float, "Weight"))

	df = df.Filter(dataframe.F{1, "Weight", "<", 260})

	df.Col("Height").Min()

	df = rescale(df, "Height")
	df = rescale(df, "Weight")

	perm := rand.Perm(df.Nrow())

	df.Subset(perm[0:int(0.7*float64(len(perm)))])

	split(df, 0.7)

	df.Col("Position")
	UniqueValues(df, "Position")
	ohSeries := OneHotSeries(df, "Position", UniqueValues(df, "Position"))
	dfEncoded := df.Mutate(ohSeries[0])
	for i := 1; i < len(ohSeries); i++ {
		dfEncoded = dfEncoded.Mutate(ohSeries[i])
	}

	dfEncoded = dfEncoded.Drop("Position")

}

//  rescale maps the given column values onto the range [0,1]
func rescale(df dataframe.DataFrame, col string) dataframe.DataFrame {
	s := df.Col(col)
	min := s.Min()
	max := s.Max()
	v := make([]float64, s.Len(), s.Len())
	for i := 0; i < s.Len(); i++ {
		v[i] = (s.Elem(i).Float() - min) / (max - min)
	}
	rs := series.Floats(v)
	rs.Name = col
	return df.Mutate(rs)
}

//  meanNormalise maps the given column values onto the range [-1,1] by subtracting mean and dividing by max - min
func meanNormalise(df dataframe.DataFrame, col string) dataframe.DataFrame {
	s := df.Col(col)
	min := s.Min()
	max := s.Max()
	mean := s.Mean()
	v := make([]float64, s.Len(), s.Len())
	for i := 0; i < s.Len(); i++ {
		v[i] = (s.Elem(i).Float() - mean) / (max - min)
	}
	rs := series.Floats(v)
	rs.Name = col
	return df.Mutate(rs)
}

//  meanNormalise maps the given column values onto the range [-1,1] by subtracting mean and dividing by max - min
func standardise(df dataframe.DataFrame, col string) dataframe.DataFrame {
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

//  split splits the dataframe into training and validation subsets. valFraction (0 <= valFraction <= 1) of the samples
//  are reserved for validation and the rest are for training.
func split(df dataframe.DataFrame, valFraction float64) (training dataframe.DataFrame, validation dataframe.DataFrame) {
	perm := rand.Perm(df.Nrow())
	cutoff := int(valFraction * float64(len(perm)))
	training = df.Subset(perm[:cutoff])
	validation = df.Subset(perm[cutoff:len(perm)])
	return training, validation
}

func UniqueValues(df dataframe.DataFrame, col string) []string {
	var ret []string
	m := make(map[string]bool)
	for _, val := range df.Col(col).Records() {
		m[val] = true
	}
	for key := range m {
		ret = append(ret, key)
	}
	return ret
}

func OneHotSeries(df dataframe.DataFrame, col string, vals []string) []series.Series {
	m := make(map[string]int)
	s := make([]series.Series, len(vals), len(vals))
	//cache the mapping for performance reasons
	for i := range vals {
		m[vals[i]] = i
	}
	for i := range s {
		vals := make([]int, df.Col(col).Len(), df.Col(col).Len())
		for j, val := range df.Col(col).Records() {
			if i == m[val] {
				vals[j] = 1
			}
		}
		s[i] = series.Ints(vals)
	}
	for i := range vals {
		s[i].Name = vals[i]
	}
	return s
}
