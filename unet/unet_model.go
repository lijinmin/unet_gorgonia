package unet

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Unet struct {
	g                          *G.ExprGraph
	n_channels                 int
	n_classes                  int
	bilinear                   bool
	down1, down2, down3, down4 struct {
		doubleConv doubleConv
	}
	up1, up2, up3, up4, outc *G.Node
	inc                      inc
}
type up struct {
}
type down struct{}
type inc struct {
	doubleConv doubleConv
}

type doubleConv struct {
	conv1      *G.Node
	conv2      *G.Node
	batchNorm1 batchNorm
	batchNorm2 batchNorm
}

// 批量归一化
type batchNorm struct {
	scale    *G.Node // 可学习参数 缩放
	bias     *G.Node // 可学习参数 偏移
	momentum float64 //
	epsilon  float64 //
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
		inc: inc{
			doubleConv: doubleConv{
				conv1: G.NewTensor(g, dt, 4, G.WithShape(64, 3, 3, 3), G.WithName("inc_doubleConv_conv1"), G.WithInit(G.GlorotN(1.0))), // output_channels=64,input_channels=3
				conv2: G.NewTensor(g, dt, 4, G.WithShape(128, 64, 3, 3), G.WithName("inc_doubleConv_conv2"), G.WithInit(G.GlorotN(1.0))),
				batchNorm1: batchNorm{
					scale:    G.NewTensor(g, dt, 1, G.WithShape(64), G.WithName("inc_doubleConv_batchNorm1_scale"), G.WithInit(G.Ones())),  // 每个通道一组数据 scale 初始化为1
					bias:     G.NewTensor(g, dt, 1, G.WithShape(64), G.WithName("inc_doubleConv_batchNorm1_bias"), G.WithInit(G.Zeroes())), // 每个通道一组数据 bias初始化为0
					momentum: 0.1,
					epsilon:  1e-5,
				},
				batchNorm2: batchNorm{
					scale:    G.NewTensor(g, dt, 1, G.WithShape(128), G.WithName("inc_doubleConv_batchNorm2_scale"), G.WithInit(G.Ones())),  // 每个通道一组数据 scale 初始化为1
					bias:     G.NewTensor(g, dt, 1, G.WithShape(128), G.WithName("inc_doubleConv_batchNorm2_bias"), G.WithInit(G.Zeroes())), // 每个通道一组数据 bias初始化为0
					momentum: 0.1,
					epsilon:  1e-5,
				},
			},
		},
	}
}

func (n *Unet) forward(x *G.Node) error {
	return nil
}
