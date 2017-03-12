package main

import (
	"math/rand"
	"github.com/gonum/matrix/mat64"
	"fmt"
	"log"
	"github.com/stillwater-sc/gokl"
	"github.com/stillwater-sc/gokl/dfablas"
)

func main() {
	// set up a CPU-side matmul operators
	A, B, C := CPUSideMatmul()

	// pretty print the data structures for visual inspection
	log.Printf("%s\n", PrettyPrintMatrix("A", A))
	log.Printf("%s\n", PrettyPrintMatrix("B", B))
	log.Printf("%s\n", PrettyPrintMatrix("C", C))


	// Now do the same operator on a remote KPU

	// Initialize creates a new resource manager
	// Does the RM work on one, or more accelerators? Multiple: the RM is the big aggregator
	// That implies that you need to get a specific KPU from the RM
	// The RM then needs to gather the KPU it manages and present that information
	// to the application.
	// TODO: how do multiple applications run concurrently?
	// If you have application space RMs then at the time of initialization the application
	// gets a snapshot of the resource state. The RM should provide the theoretical resource limits
	// and the semantcs of resource allocation/deallocation will be done in the context of the state
	// available at the time the application calls for the allocation. This would make it possible
	// for two or more applications to be active at the same time. If one application asks for
	// too much resource, it would receive an error, and it will need to adapt accordingly.
	rm := gokl.Initialize()
	kpu := rm.GetRemote("KPU-0")
	defer func() {
		gokl.Shutdown()
	}()

	// marshal the A and B matrices into the KPU fabric
	var targets []*gokl.KPU = make([]*gokl.KPU, 1)
	targets[0] = kpu
	Akpu, err := rm.Marshal(A, targets)		// marshal a CPU data structure to a set of target KPUs
	if err != nil {
		log.Fatal(err)
	}
	Bkpu, err := rm.Marshal(B, targets)
	if err != nil {
		log.Fatal(err)
	}
	// Create the zero type input C matrix
	var Cdescriptor gokl.DataStructureDescriptor = gokl.DataStructureDescriptor{Form:gokl.KL_DENSE_MATRIX, Elements:gokl.KL_FLOAT64, Bits:64, N:5, M:5, K:0}
	Ckpu, err := rm.New(Cdescriptor)
	if err != nil {
		log.Fatal(err)
	}
	// Create the output D matrix to receive the results from the domain flow computation
	var Ddescriptor gokl.DataStructureDescriptor = gokl.DataStructureDescriptor{Form:gokl.KL_DENSE_MATRIX, Elements:gokl.KL_FLOAT64, Bits:64, N:5, M:5, K:0}
	Dkpu, err := rm.New(Ddescriptor)
	if err != nil {
		log.Fatal(err)
	}
	// now all the input and outputs have been defined, allocated, and initialized, call the operator chain
	dfablas.GEMM(Akpu,Bkpu, Ckpu, Dkpu)
	// marshal the result data set back to the CPU address space
}

func CPUSideMatmul() (A,B,Cout *mat64.Dense) {
	// create and initialize the input matrices
	A = mat64.NewDense(5,5, CreateRandom(25))
	B = mat64.NewDense(5,5, CreateRandom(25))
	// define a zero size output matrix to receive the result
	var C mat64.Dense
	C.Mul(A,B)
	Cout = &C
	return
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
