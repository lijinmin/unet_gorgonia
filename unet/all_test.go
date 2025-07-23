package unet

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"testing"
)

func TestConv(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	yData := []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData)) // batch channel height width
	yVal := tensor.New(tensor.WithShape(4, 1, 2, 2), tensor.WithBacking(yData)) // output_channel input_channel height width
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))
	y := G.NewTensor(g, tensor.Float64, 4, G.WithValue(yVal), G.WithName("y"))
	z := G.Must(G.Conv2d(x, y, tensor.Shape{2, 2}, []int{0, 0}, []int{1, 1}, []int{2, 2})) // 一般用来处理图像,y的每个input_channel的核分别与x的每个channel相乘，然后再求和

	vm := G.NewTapeMachine(g)
	defer vm.Close()
	t.Log(y.Value())
	for i := 0; i < 1; i++ {
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}

		//xData = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
		//xVal = tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(xData))
		//G.Let(x, xVal)

		t.Log(z.Shape())
		t.Log(z.Value())
		vm.Reset()
	}
}
