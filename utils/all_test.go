package utils

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"testing"
)

func TestPad(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData)) // batch channel height width
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))

	t.Log(3/2, 1/2)
	z, err := G.Pad(x, []int{1, 1, 1, 1})
	if err != nil {
		t.Log(err)
	}
	t.Log(z.Shape())
	t.Log(x.Value())
	vm := G.NewTapeMachine(g)
	defer vm.Close()
	for i := 0; i < 1; i++ {
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}

		//xData = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
		//xVal = tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(xData))
		//G.Let(x, xVal)
		t.Log(z.Value())
		vm.Reset()
	}
}

func TestConCat(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData)) // batch channel height width
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))

	yData := []float64{16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}
	yVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(yData)) // batch channel height width
	y := G.NewTensor(g, tensor.Float64, 4, G.WithValue(yVal), G.WithName("y"))

	t.Log(3/2, 1/2)
	z, err := G.Concat(1, y, x)
	if err != nil {
		t.Log(err)
	}
	t.Log(z.Shape())
	t.Log(x.Value())
	vm := G.NewTapeMachine(g)
	defer vm.Close()
	for i := 0; i < 1; i++ {
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}

		//xData = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
		//xVal = tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(xData))
		//G.Let(x, xVal)
		t.Log(z.Value(), z.Shape())
		vm.Reset()
	}
}
func TestMaxPool2D(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData))
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal))

	//一般用于卷积神经网络空间下采样
	z := G.Must(G.MaxPool2D(x, tensor.Shape{2, 2}, []int{1, 1}, []int{1, 1})) // kernel选取池大小，pad 输入矩阵四周补0层数，stride 选取池移动速度,每次都是从选取池里取出最大值作为当前元素的值

	vm := G.NewTapeMachine(g)
	defer vm.Close()

	t.Log(x.Node().ID(), z.Node().ID())

	for i := 0; i < 1; i++ {
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}
		t.Log("dddddddd")
		//xData = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
		//xVal = tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(xData))
		//G.Let(x, xVal)
		t.Log("\n", x.Value())
		t.Log("\n", z.Value(), z.Shape())
		vm.Reset()
	}
}
