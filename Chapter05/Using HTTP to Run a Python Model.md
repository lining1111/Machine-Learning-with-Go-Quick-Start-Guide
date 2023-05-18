```go
import (
    "fmt"
     mnist "github.com/petar/GoMNIST"
    "github.com/kniren/gota/dataframe"
    "github.com/kniren/gota/series"
    "image"
    "bytes"
    "math"
)
```


```go
set, err := mnist.ReadSet("../datasets/mnist/images.gz", "../datasets/mnist/labels.gz")
```


```go
set.Images[1]
```




    
![png](Using%20HTTP%20to%20Run%20a%20Python%20Model_files/Using%20HTTP%20to%20Run%20a%20Python%20Model_2_0.png)
    




```go
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
```


```go
df := MNISTSetToDataframe(set, 1000)
```


```go
categories := []string{"tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "shoe", "bag", "boot"}
```


```go
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
```


```go
testImages := ImageSeriesToInts(df, "Image")
```

## Invoke the model using JSON-RPC


```go
import (
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"time"
    "io/ioutil"
    "encoding/json"
    "bytes"
)
```


```go
c, err := jsonrpc.Dial("tcp", "localhost:8001")
p := model{Client: client}
var req PredictRequest = PredictRequest{
        Image: testImages[16],
}

var reply interface{}
err := c.Call("predict", req, &reply)
```


```go
//  Predict returns whether the ith image represents trousers or not based on the logistic regression model
func Predict(i int) (bool, error){
    b, err := json.Marshal(testImages[i])
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
    var resp struct {
        IsTrousers bool `json:"is_trousers"`
    }
    err := json.Unmarshal(body, &resp)
    return resp.IsTrousers, err    
}

```


```go
// Expected: true <nil>
Predict(16)
```




    true <nil>




```go
// Expected false <nil>
Predict(0)
```




    false <nil>




```go

```
