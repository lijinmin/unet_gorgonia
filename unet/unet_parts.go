package unet

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func DoubleConv(x *G.Node, dc *doubleConv) (*G.Node, error) {
	//retVal1, err := G.Conv2d(x, dc.conv1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	//if err != nil {
	//	errors.Wrap(err, "DoubleConv 1 Convolution failed")
	//}
	//retVal2, _, err := BatchNorm(retVal1, dc.batchNorm1)
	retVal2, _, err := BatchNorm(G.Must(G.Conv2d(x, dc.conv1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})), dc.batchNorm1)
	if err != nil {
		return retVal2, err
	}
	//retVal3, err := G.Rectify(retVal2)
	//if err != nil {
	//	return retVal3, err
	//}

	//retVal4, err := G.Conv2d(retVal3, dc.conv2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	//if err != nil {
	//	return retVal4, err
	//}
	//retVal5, _, err := BatchNorm(retVal4, dc.batchNorm2)
	retVal5, _, err := BatchNorm(G.Must(G.Conv2d(G.Must(G.Rectify(retVal2)), dc.conv2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})), dc.batchNorm2)
	if err != nil {
		return retVal5, err
	}
	retVal6, err := G.Rectify(retVal5)
	if err != nil {
		return retVal6, err
	}
	return retVal6, err
}

// 归一化 加快训练
func BatchNorm(x *G.Node, n *batchNorm) (*G.Node, *G.BatchNormOp, error) {
	retVal, _, _, op, err := G.BatchNorm(x, n.scale, n.bias, n.momentum, n.epsilon)
	if err != nil {
		n.op = op
	}
	return retVal, op, err
}

func Down() {

}

func upsample2D_0() {

}
