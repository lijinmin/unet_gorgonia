package utils

import (
	G "gorgonia.org/gorgonia"
)

func Pad(x *G.Node) (*G.Node, error) {
	retVal, err := G.Im2Col(x, []int{1, 1}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	return retVal, err
}
