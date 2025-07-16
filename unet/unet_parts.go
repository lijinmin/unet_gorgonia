package unet

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func DoubleConv(x *G.Node, inc *doubleConv) (n *G.Node, err error) {
	n, err = G.Conv2d(x, inc.conv1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	G.BatchNorm()
	G.Rectify()
	G.BatchNormOp{}.SetStats()
	g := G.NewGraph()
	//G.EvalMode()
	//	G.TrainModeOp()
	// 训练模式（关键选项）

	G.norm
	if err != nil { /* 处理错误 */
	}
	return n, errors.Wrap(err, "Layer 0 Convolution failed")
}

func Down() {

}
