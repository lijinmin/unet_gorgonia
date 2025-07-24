package unet

import (
	"fmt"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Unet struct {
	g                          *G.ExprGraph
	n_channels                 int
	n_classes                  int
	bilinear                   bool
	down1, down2, down3, down4 *down
	up1, up2, up3, up4         *up
	outc                       *doubleConv
	inc                        *inc
}

func (u Unet) Learnables() G.Nodes {
	return G.Nodes{}
}

type up struct {
	upsample2DScale int
	filter          *G.Node
	doubleConv      *doubleConv
}
type inc struct {
	doubleConv *doubleConv
}
type down struct {
	doubleConv *doubleConv
	maxPool2D  struct {
		kernelSize int
		stride     int
		padding    int
		dilation   int
	}
}

type doubleConv struct {
	conv1      *G.Node
	conv2      *G.Node
	batchNorm1 *batchNorm
	batchNorm2 *batchNorm
}

// 批量归一化
type batchNorm struct {
	scale                        *G.Node // 可学习参数 缩放
	bias                         *G.Node // 可学习参数 偏移
	momentum                     float64 //
	epsilon                      float64 //
	op                           *G.BatchNormOp
	runningMean, runningVariance tensor.Tensor
}

// 下采样参数
type maxPool2D struct {
	kernelSize int // 池化窗口大小
	stride     int // 窗口移动步长
	padding    int // 边缘填充像素数
	dilation   int // 控制窗口内元素的间距
}

func NewUnet(g *G.ExprGraph, n_channels, n_classes int, bilinear bool, dt tensor.Dtype) *Unet {
	if n_channels == 0 {
		n_channels = 3
	}
	if n_classes == 0 {
		n_classes = 1
	}

	return &Unet{
		g:          g,
		n_channels: n_channels,
		n_classes:  n_classes,
		bilinear:   bilinear,
		inc: &inc{
			doubleConv: newDoubleConv(g, dt, n_channels, 64, 64, "inc"),
		},
		down1: newDown(g, dt, 64, 128, 128, "down1"),
		down2: newDown(g, dt, 128, 256, 256, "down2"),
		down3: newDown(g, dt, 256, 512, 512, "down3"),
		down4: newDown(g, dt, 512, 1024, 1024, "down4"),
		up1:   newUp(g, dt, 1024, 512, 512, 2, "up1"),
		up2:   newUp(g, dt, 512, 256, 256, 2, "up2"),
		up3:   newUp(g, dt, 256, 128, 128, 2, "up3"),
		up4:   newUp(g, dt, 128, 64, 64, 2, "up4"),
		outc:  newDoubleConv(g, dt, 64, n_classes, n_classes, "outc"),
	}
}

func newDown(g *G.ExprGraph, dt tensor.Dtype, inputChannels, midChannels, outputChannels int, label string) *down {
	return &down{
		doubleConv: newDoubleConv(g, dt, inputChannels, midChannels, outputChannels, label),
		maxPool2D: struct {
			kernelSize int
			stride     int
			padding    int
			dilation   int
		}{kernelSize: 2, stride: 2, padding: 0, dilation: 1},
	}
}

func newUp(g *G.ExprGraph, dt tensor.Dtype, inputChannels, midChannels, outputChannels, scale int, label string) *up {
	return &up{
		upsample2DScale: scale,
		doubleConv:      newDoubleConv(g, dt, inputChannels, midChannels, outputChannels, label),
		filter:          G.NewTensor(g, dt, 4, G.WithShape(inputChannels, outputChannels, 3, 3), G.WithName(fmt.Sprintf("%s_filter", label)), G.WithInit(G.GlorotN(1.0))),
	}
}

func newDoubleConv(g *G.ExprGraph, dt tensor.Dtype, inputChannels, midChannels, outputChannels int, label string) *doubleConv {
	if midChannels == 0 {
		midChannels = outputChannels
	}
	return &doubleConv{
		conv1: G.NewTensor(g, dt, 4, G.WithShape(midChannels, inputChannels, 3, 3), G.WithName(fmt.Sprintf("%s_doubleConv_conv1", label)), G.WithInit(G.GlorotN(1.0))), // output_channels，input_channels=
		conv2: G.NewTensor(g, dt, 4, G.WithShape(outputChannels, midChannels, 3, 3), G.WithName(fmt.Sprintf("%s_doubleConv_conv2", label)), G.WithInit(G.GlorotN(1.0))),
		batchNorm1: &batchNorm{
			scale:    G.NewTensor(g, dt, 4, G.WithShape(1, midChannels, 1, 1), G.WithName(fmt.Sprintf("%s_doubleConv_batchNorm1_scale", label)), G.WithInit(G.Ones())),  // 每个通道一组数据 scale 初始化为1
			bias:     G.NewTensor(g, dt, 4, G.WithShape(1, midChannels, 1, 1), G.WithName(fmt.Sprintf("%s_doubleConv_batchNorm1_bias", label)), G.WithInit(G.Zeroes())), // 每个通道一组数据 bias初始化为0
			momentum: 0.1,
			epsilon:  1e-5,
		},
		batchNorm2: &batchNorm{
			scale:    G.NewTensor(g, dt, 4, G.WithShape(1, outputChannels, 1, 1), G.WithName(fmt.Sprintf("%s_doubleConv_batchNorm2_scale", label)), G.WithInit(G.Ones())),  // 每个通道一组数据 scale 初始化为1
			bias:     G.NewTensor(g, dt, 4, G.WithShape(1, outputChannels, 1, 1), G.WithName(fmt.Sprintf("%s_doubleConv_batchNorm2_bias", label)), G.WithInit(G.Zeroes())), // 每个通道一组数据 bias初始化为0
			momentum: 0.1,                                                                                                                                                  //对小数据集或动态数据：用较小的 momentum（如 0.1）;对大数据集或稳定数据：用较大的 momentum（如 0.9）
			epsilon:  1e-5,
		},
	}
}
func (n *inc) forward(x *G.Node) (*G.Node, error) {
	retVal, err := DoubleConv(x, n.doubleConv)
	return retVal, err
}

func (d *down) forward(x *G.Node) (*G.Node, error) {
	retVal, err := G.MaxPool2D(x, tensor.Shape{d.maxPool2D.kernelSize, d.maxPool2D.kernelSize}, []int{0, 0}, []int{d.maxPool2D.stride, d.maxPool2D.stride})
	if err != nil {
		return retVal, err
	}
	retVal1, err := DoubleConv(retVal, d.doubleConv)
	return retVal1, err
}

func (u *up) forward(x1, x2 *G.Node) (*G.Node, error) {
	retVal0, err := G.Upsample2D(x1, 2) // 需调整
	if err != nil {
		return retVal0, err
	}
	retVal1, err := G.Conv2d(retVal0, u.filter, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}, []int{1, 1})
	if err != nil {
		return nil, err
	}

	diffY := x2.Shape()[2] - retVal1.Shape()[2]
	diffX := x2.Shape()[3] - retVal1.Shape()[3]
	retVal2, err := G.Pad(retVal1, []int{diffY / 2, diffY - diffY/2, diffX / 2, diffX - diffX/2})
	if err != nil {
		return nil, err
	}
	retVal3, err := G.Concat(1, x2, retVal2)
	if err != nil {
		return nil, err
	}

	retVal4, err := DoubleConv(retVal3, u.doubleConv)
	return retVal4, err

}

func (n *Unet) Forward(x *G.Node) (*G.Node, error) {
	retVal1, err := n.inc.forward(x)
	if err != nil {
		return nil, err
	}

	retVal2, err := n.down1.forward(retVal1)
	if err != nil {
		return nil, err
	}

	retVal3, err := n.down2.forward(retVal2)
	if err != nil {
		return nil, err
	}

	retVal4, err := n.down3.forward(retVal3)
	if err != nil {
		return nil, err
	}

	retVal5, err := n.down4.forward(retVal4)
	if err != nil {
		return nil, err
	}

	retVal6, err := n.up1.forward(retVal5, retVal4)
	if err != nil {
		return nil, err
	}

	retVal7, err := n.up2.forward(retVal6, retVal3)
	if err != nil {
		return nil, err
	}

	retVal8, err := n.up3.forward(retVal7, retVal2)
	if err != nil {
		return nil, err
	}

	retVal9, err := n.up4.forward(retVal8, retVal1)
	if err != nil {
		return nil, err
	}

	retVal10, err := DoubleConv(retVal9, n.outc)

	return retVal10, err
}
