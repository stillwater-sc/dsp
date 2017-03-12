package main

import (
	"math/rand"
	"github.com/gonum/matrix/mat64"
	"fmt"
	"log"
)

func main() {
	a_values := make([]float64, 25)
	b_values := make([]float64, 25)
	for i := range a_values {
		a_values[i] = rand.NormFloat64()
		b_values[i] = rand.NormFloat64()
	}
	A := mat64.NewDense(5,5, a_values)
	B := mat64.NewDense(5,5, b_values)

	var C mat64.Dense
	C.Mul(A,B)
	rows, cols := C.Dims()
	c_values := make([]float64, 25)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			c_values[i * cols + j] = C.At(i, j)
		}
	}
	log.Printf("%s\n", PrettyPrintMatrix("A", 5,5, a_values))
	log.Printf("%s\n", PrettyPrintMatrix("B", 5,5, b_values))
	log.Printf("%s\n", PrettyPrintMatrix("C", 5,5, c_values))
}

func PrettyPrintMatrix(name string, rows, cols int, data []float64) string {
	stride := cols
	// create header
	var str string = fmt.Sprintf("\n%10s", name)
	for j := 0; j < rows; j++ {
		str = str + fmt.Sprintf("%10d ", j)
	}
	str = str + "\n"
	for i := 0; i < rows; i++ {
		str = str + fmt.Sprintf("%10d ", i)
		for j := 0; j < cols; j++ {
			str = str + fmt.Sprintf("%10.6f ", data[i*stride + j])
		}
		str = str + "\n"
	}
	return str
}
