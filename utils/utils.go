package utils

import (
	"fmt"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
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

		if formPadding[i] < 0 {
			ss := make([]tensor.Slice, dims)
			for j := 0; j < dims; j++ {
				if j == dim {
					if i%2 == 0 {
						ss[i] = G.S(-1*formPadding[i], shape[dim])
					} else {
						ss[i] = G.S(0, shape[dim]+formPadding[i])
					}

				} else {
					ss[i] = G.S(0, shape[dim])
				}
			}
			result, _ = G.Slice(input, ss...)

			//nodes, _ := G.Unconcat(result, dim, -1*formPadding[i])
			//if i%2 == 0 {
			//	result = nodes[1]
			//} else {
			//	result = nodes[0]
			//}

		} else {
			shape[dim] = formPadding[i]
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
