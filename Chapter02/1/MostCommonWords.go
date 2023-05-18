package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"io/ioutil"
	"strconv"
	"strings"
)

var kitchenReviews = "../datasets/words/processed_acl/kitchen"

func main() {

	positives, err := ioutil.ReadFile(kitchenReviews + "/positive.review")
	negatives, err2 := ioutil.ReadFile(kitchenReviews + "/negative.review")
	if err != nil || err2 != nil {
		fmt.Println("Error(s)", err, err2)
	}

	//string(positives)[0:100]

	pairsPositive := strings.Split(strings.Replace(string(positives), "\n", " ", -1), " ")
	pairsNegative := strings.Split(strings.Replace(string(negatives), "\n", " ", -1), " ")

	fmt.Println("Positive pairs", len(pairsPositive))
	fmt.Println("Negative Pairs", len(pairsPositive))
	fmt.Printf("Example pair: `%s`", pairsPositive[0])

	parsedPositives, posPhrases := pairsAndFilters(pairsPositive)
	parsedNegatives, negPhrases := pairsAndFilters(pairsNegative)
	parsedPositives = exclude(parsedPositives, negPhrases)
	parsedNegatives = exclude(parsedNegatives, posPhrases)

	dfPos := dataframe.LoadStructs(parsedPositives)
	dfNeg := dataframe.LoadStructs(parsedNegatives)

	dfPos = dfPos.Arrange(dataframe.RevSort("Frequency"))
	dfNeg = dfNeg.Arrange(dataframe.RevSort("Frequency"))

	//most common words in positive reviews
	fmt.Println(dfPos)

	// most common words in negative reviews
	fmt.Println(dfNeg)

}

type Pair struct {
	Phrase    string
	Frequency int
}

//  pairsAndFiltesr returns a slice of Pair, split by : to obtain the phrase and frequency,
//  as well as a map of the phrases that can be used as a lookup table later.
func pairsAndFilters(splitPairs []string) ([]Pair, map[string]bool) {
	var (
		pairs []Pair
		m     map[string]bool
	)
	m = make(map[string]bool)
	for _, pair := range splitPairs {
		p := strings.Split(pair, ":")
		phrase := p[0]
		m[phrase] = true
		if len(p) < 2 {
			continue
		}
		freq, err := strconv.Atoi(p[1])
		if err != nil {
			continue
		}
		pairs = append(pairs, Pair{
			Phrase:    phrase,
			Frequency: freq,
		})
	}
	return pairs, m
}

//  exclude returns a slice of Pair that does not contain the phrases in the exclusion map
func exclude(pairs []Pair, exclusions map[string]bool) []Pair {
	var ret []Pair
	for i := range pairs {
		if !exclusions[pairs[i].Phrase] {
			ret = append(ret, pairs[i])
		}
	}
	return ret
}
