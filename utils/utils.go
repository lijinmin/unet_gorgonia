package utils

import (
	"fmt"
	G "gorgonia.org/gorgonia"
)

func Pad(g *G.ExprGraph, input *G.Node, padding []int, mode string) (*G.Node, error) {
	if len(padding) != 2 && len(padding) != 4 {
		return nil, fmt.Errorf("padding length must be 2 or 4")
	}
	// 默认使用零填充
	if mode == "" {
		mode = "constant"
	}

	var result *G.Node = input
	var padTop, padBottom, padLeft, padRight int
	dims := input.Dims()
	formPadding := []int{}

	if len(padding) == 2 {
		padTop = padding[1]
		padBottom = padding[1]
		padLeft = padding[0]
		padRight = padding[0]
	} else {
		padLeft = padding[0]
		padRight = padding[1]
		padTop = padding[2]
		padBottom = padding[3]
	}
	formPadding = append(formPadding, padLeft, padRight, padTop, padBottom)
	for i := 0; i < 4; i++ {
		if formPadding[i] == 0 {
			continue
		}
		dim := dims - 1 - i/2 // 维度
		shape := result.Shape()
		shape[dim] = formPadding[i]
		if formPadding[i] < 0 {
			nodes, _ := G.Unconcat(result, dim, -1*formPadding[i])
			result = nodes[0]
		} else {
			zeros := G.NewTensor(g, input.Dtype(), dims, G.WithShape(shape...), G.WithInit(G.Zeroes()))
			if i%2 == 0 {
				result, _ = G.Concat(dim, zeros, result)
			} else {
				result, _ = G.Concat(dim, result, zeros)
			}
		}

	}

	return result, nil
}
