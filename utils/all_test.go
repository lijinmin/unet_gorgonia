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
	xData := []float64{5, 2, 10, 2, 1, 2, 4, 4}
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

func TestLog(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{5, 2, 1e1, 2, 1, 10, 4, 4}                               // 1e1 = 10
	xVal := tensor.New(tensor.WithShape(1, 2, 1, 4), tensor.WithBacking(xData)) //

	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal))
	z := G.Must(G.Log(x))  // ln(x)
	n := G.Must(G.Log2(x)) // log(x) 以2为低
	vm := G.NewTapeMachine(g)
	defer vm.Close()
	vm.RunAll()
	t.Log("\n", x.Value())
	t.Log(z.Value())
	t.Log(n.Value())
}

func TestMean(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{5, 2, 1, 2, 1, 2, 4, 4, 5, 2, 1, 2, 1, 2, 4, 4}
	xVal := tensor.New(tensor.WithShape(2, 2, 1, 4), tensor.WithBacking(xData)) //

	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal))
	z := G.Must(G.SoftMax(x, 1))

	d, err := G.Log2(z)
	if err != nil {
		t.Log(err)
	}

	n, err := G.Mean(x)
	t.Log(n.Shape())
	vm := G.NewTapeMachine(g)
	defer vm.Close()
	vm.RunAll()
	t.Log("\n", x.Value())
	t.Log(z.Value())
	t.Log(d.Value())
	t.Log(n.Value())
}

func TestSum(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{5, 2, 1, 2, 3, 2, 4, 4, 5, 2, 1, 2, 1, 2, 4, 4}
	xVal := tensor.New(tensor.WithShape(2, 2, 1, 4), tensor.WithBacking(xData)) //

	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal))

	z := G.Must(G.Sum(x, 1, 3))

	vm := G.NewTapeMachine(g)
	defer vm.Close()
	vm.RunAll()
	t.Log("\n", x.Value())
	t.Log(z.Value())

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
func TestPad2(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData)) // batch channel height width
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))

	t.Log(3/2, 1/2)
	z, err := Pad(g, x, []int{-1, 1, -2, 2}, "")
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
		t.Log(z.Shape())
		vm.Reset()
	}
}
func TestUnContat(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData)) // batch channel height width
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))

	t.Log(3/2, 1/2)
	nodes, err := G.Unconcat(x, 3, 2)
	if err != nil {
		t.Log(err)
	}
	t.Log(len(nodes))
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
		t.Log(nodes[0].Value())
		t.Log(nodes[1].Value())
		t.Log(nodes[0].Shape())
		vm.Reset()
	}
}

func TestSlice(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	xVal := tensor.New(tensor.WithShape(1, 1, 4, 4), tensor.WithBacking(xData)) // batch channel height width
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))

	t.Log(3/2, 1/2)
	z, err := G.Slice(x, G.S(0, 1), G.S(0, 1), G.S(0, 4), G.S(0, 3))
	if err != nil {
		t.Log(err)
	}
	t.Log(z.Shape())
	d, _ := G.Reshape(z, tensor.Shape{1, 1, 4, 3})
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
		t.Log(z.Shape())
		t.Log(d.Value())
		t.Log(d.Shape())
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

func TestHadamardDiv(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	yData := []float64{2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 5, 1, 2, 1, 2, 1, 2, 1, 1, 3, 6, 1, 5, 1, 2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 5, 1, 2, 1, 2, 1, 2, 1, 1, 3, 6, 1, 5, 1}
	xVal := tensor.New(tensor.WithShape(2, 2, 3, 4), tensor.WithBacking(xData)) //
	yVal := tensor.New(tensor.WithShape(2, 2, 3, 4), tensor.WithBacking(yData))
	t.Log(xVal)
	t.Log(yVal)
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))
	y := G.NewTensor(g, tensor.Float64, 4, G.WithValue(yVal), G.WithName("y"))
	z := G.Must(G.HadamardDiv(G.Must(G.Sum(x, 1, 2, 3)), G.Must(G.Sum(y, 1, 2, 3))))
	z = G.Must(G.Mul(G.NewConstant(2.0), z))
	vm := G.NewTapeMachine(g)
	defer vm.Close()

	vm.RunAll()
	t.Log("\n", x.Value())
	t.Log("\n", y.Value())
	t.Log("\n", z.Value(), z.Shape())
}

func TestConstant(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	yData := []float64{2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 5, 1, 2, 1, 2, 1, 2, 1, 1, 3, 6, 1, 5, 1}
	xVal := tensor.New(tensor.WithShape(1, 2, 3, 4), tensor.WithBacking(xData)) //
	yVal := tensor.New(tensor.WithShape(1, 2, 3, 4), tensor.WithBacking(yData))
	t.Log(xVal)
	t.Log(yVal)
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))
	y := G.NewTensor(g, tensor.Float64, 4, G.WithValue(yVal), G.WithName("y"))
	z := G.Must(G.HadamardDiv(x, y))

	nn := G.Must(G.Mul(G.NewConstant(1.0), G.Must(G.Mean(z))))

	zz := G.Must(G.Sub(G.NewConstant(1.0), G.NewConstant(0.5)))
	vm := G.NewTapeMachine(g)

	defer vm.Close()

	vm.RunAll()
	t.Log("\n", x.Value())
	t.Log("\n", y.Value())
	t.Log("\n", z.Value())
	t.Log(nn.Value())
	t.Log(zz.Value())
}

func TestGrad(t *testing.T) {
	g := G.NewGraph()
	xData := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	yData := []float64{2, 1, 2, 1, 2, 1, 1, 3, 1, 1, 5, 1, 2, 1, 2, 1, 2, 1, 1, 3, 6, 1, 5, 1}
	xVal := tensor.New(tensor.WithShape(1, 2, 3, 4), tensor.WithBacking(xData)) //
	yVal := tensor.New(tensor.WithShape(1, 2, 3, 4), tensor.WithBacking(yData))
	t.Log(xVal)
	t.Log(yVal)
	x := G.NewTensor(g, tensor.Float64, 4, G.WithValue(xVal), G.WithName("x"))
	y := G.NewTensor(g, tensor.Float64, 4, G.WithValue(yVal), G.WithName("y"))
	z := G.Must(G.HadamardDiv(x, y))

	nn := G.Must(G.Mul(G.NewConstant(1.0), G.Must(G.Mean(z))))

	nodes := G.Nodes{G.NewTensor(g, tensor.Float64, 4, G.WithShape(32, 1, 3, 3), G.WithName("w0"), G.WithInit(G.GlorotN(1.0)))}
	cost := G.Must(G.Mean(nn))
	G.Grad(cost, nodes...)

	vm := G.NewTapeMachine(g)

	defer vm.Close()

	vm.RunAll()
	t.Log("\n", x.Value())
	t.Log("\n", y.Value())
	t.Log("\n", z.Value())
	t.Log(nn.Value())
}

func TestShuffle(t *testing.T) {
	a := []string{"1", "2", "3", "4", "5", "6", "7", "8", "9"}
	t.Log(a)
	Shuffle(a)
	t.Log(a)
}

func TestLoadImage(t *testing.T) {
	s := "../data/imgs/fff9b3a5373f_07.jpg"
	//s1 := "../data/masks/fff9b3a5373f_07_mask.gif"
	img, err := loadImage(s)
	if err != nil {
		t.Log(err)
	}
	rect := img.Bounds()
	minPoint := rect.Min
	maxPoint := rect.Max
	for i := minPoint.X; i < maxPoint.X; i++ {
		for j := minPoint.Y; j < maxPoint.Y; j++ {
			img.At(i, j)
		}
	}

	t.Log(rect, maxPoint.X)

}
