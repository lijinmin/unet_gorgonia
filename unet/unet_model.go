package unet

import (
	G "gorgonia.org/gorgonia"
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
}

// 下采样参数
type maxPool2D struct {
	kernelSize int // 池化窗口大小
	stride     int // 窗口移动步长
	padding    int // 边缘填充像素数
	dilation   int // 控制窗口内元素的间距
}

func NewUnet(g *G.ExprGraph, n_channels, n_classes int, bilinear bool) *Unet {
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
			doubleConv: doubleConv{},
		},
	}
}

func (n *Unet) forward(x *G.Node) error {
	return nil
}
