package main

import (
	"math/rand"
	"github.com/gonum/matrix/mat64"
	"fmt"
	"log"
	"github.com/stillwater-sc/gokl"
)

func main() {
	accelerator := gokl.Initialize()
	defer func() {
		accelerator.Release()
	}()

	A := mat64.NewDense(5,5, CreateRandom(25))
	B := mat64.NewDense(5,5, CreateRandom(25))

	var C mat64.Dense
	C.Mul(A,B)

	log.Printf("%s\n", PrettyPrintMatrix("A", A))
	log.Printf("%s\n", PrettyPrintMatrix("B", B))
	log.Printf("%s\n", PrettyPrintMatrix("C", &C))
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
