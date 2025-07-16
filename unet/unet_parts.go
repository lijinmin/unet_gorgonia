package unet

import (
	"github.com/pkg/errors"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func DoubleConv(x *G.Node, inc *doubleConv) (retVal *G.Node, err error) {
	retVal, err = G.Conv2d(x, inc.conv1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		errors.Wrap(err, "DoubleConv 1 Convolution failed")
	}
	retVal, op, err := BatchNorm(retVal, inc.batchNorm1)
	return retVal, errors.Wrap(err, "Layer 0 Convolution failed")
}

func BatchNorm(x *G.Node, n batchNorm) (*G.Node, *G.BatchNormOp, error) {
	retVal, _, _, op, err := G.BatchNorm(x, n.scale, n.bias, n.momentum, n.epsilon)
	return retVal, op, err
}

func Down() {

}
