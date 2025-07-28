package utils

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"testing"
)

func TestConv2d(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	yData := []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 1, 2}
	//aData := []float64{0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData)) // batch channel height width
	yVal := tensor.New(tensor.WithShape(4, 1, 3, 3), tensor.WithBacking(yData)) // output_channel input_channel height width
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))
	y := G.NewTensor(g, tensor.Float64, 4, G.WithValue(yVal), G.WithName("y"))
	z := G.Must(G.Conv2d(x, y, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})) // 一般用来处理图像,y的每个input_channel的核分别与x的每个channel相乘，然后再求和

	t.Log(z.Shape())

	//d := G.Must(G.Conv2d(z, a, tensor.Shape{2, 2}, []int{1, 1}, []int{1, 1}, []int{1, 1}))
	vm := G.NewTapeMachine(g)
	//t.Log(a.Value(), a.Shape())
	defer vm.Close()

	for i := 0; i < 1; i++ {
		if err := vm.RunAll(); err != nil {
			t.Fatal(err)
		}

		//xData = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
		//xVal = tensor.New(tensor.WithShape(3, 3), tensor.WithBacking(xData))
		//G.Let(x, xVal)

		t.Log(z.Shape())
		//t.Log(a.Value(), a.Shape())
		//t.Log(d.Shape())
		vm.Reset()
	}
}

func TestSoftMax(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{5, 2, 1, 2, 1, 2, 4, 4}
	xVal := tensor.New(tensor.WithShape(1, 2, 1, 4), tensor.WithBacking(xData)) //

	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal))
	z := G.Must(G.SoftMax(x, 1))

	d, err := G.Log2(z)
	if err != nil {
		t.Log(err)
	}
	vm := G.NewTapeMachine(g)
	defer vm.Close()
	vm.RunAll()
	t.Log("\n", x.Value())
	t.Log(z.Value())
	t.Log(d.Value())
}

func TestTensor(t *testing.T) {
	xData := []float64{1, 2, 3, 4, 1, 2, 3, 4}
	xVal := tensor.New(tensor.WithShape(1, 2, 1, 4), tensor.WithBacking(xData)) //
	xVal.Set(3, float64(6))
	t.Log(xVal)
}

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

func TestHadamardProd(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	yData := []float64{2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 5, 1, 2, 1, 2, 1, 2, 1, 1, 3, 6, 1, 5, 1}
	xVal := tensor.New(tensor.WithShape(1, 2, 3, 4), tensor.WithBacking(xData)) //
	yVal := tensor.New(tensor.WithShape(1, 2, 3, 4), tensor.WithBacking(yData))
	t.Log(xVal)
	t.Log(yVal)
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))
	y := G.NewTensor(g, tensor.Float64, 4, G.WithValue(yVal), G.WithName("y"))
	z := G.Must(G.HadamardProd(x, y))
	vm := G.NewTapeMachine(g)
	defer vm.Close()

	vm.RunAll()
	t.Log("\n", x.Value())
	t.Log("\n", y.Value())
	t.Log("\n", z.Value())
}
