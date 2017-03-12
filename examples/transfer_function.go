package main

import (
	"math/rand"
	"github.com/gonum/matrix/mat64"
	"fmt"
	"log"
	"github.com/stillwater-sc/gokl"
)

func main() {
	// Initialize creates a new resource manager
	// Does the RM work on one, or more accelerators? Multiple: the RM is the big aggregator
	// That implies that you need to get a specific KPU from the RM
	rm := gokl.Initialize()
	kpu := rm.GetRemote("KPU-0")
	defer func() {
		gokl.Shutdown()
	}()

	A := mat64.NewDense(5,5, CreateRandom(25))
	B := mat64.NewDense(5,5, CreateRandom(25))

	var C mat64.Dense
	C.Mul(A,B)

	log.Printf("%s\n", PrettyPrintMatrix("A", A))
	log.Printf("%s\n", PrettyPrintMatrix("B", B))
	log.Printf("%s\n", PrettyPrintMatrix("C", &C))

	// marshal the A and B matrices into the KPU fabric
	rm.Marshal(A)
}

func CreateRandom(size int) []float64 {
	values := make([]float64, size)
	for i := range values {
		values[i] = rand.NormFloat64()
	}
	return values
}

func PrettyPrintMatrix(name string, matrix *mat64.Dense) string {
	rows, cols := matrix.Dims()
	// create header
	var str string = fmt.Sprintf("\n%10s", name)
	for j := 0; j < rows; j++ {
		str = str + fmt.Sprintf("%10d ", j)
	}
	str = str + "\n"
	for i := 0; i < rows; i++ {
		str = str + fmt.Sprintf("%10d ", i)
		for j := 0; j < cols; j++ {
			str = str + fmt.Sprintf("%10.6f ", matrix.At(i,j))
		}
		str = str + "\n"
	}
	return str
}
