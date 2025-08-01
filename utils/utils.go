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
		if formPadding[i] <= 0 {
			continue
		}
		dim := dims - 1 - i/2 // 维度
		zeroshape := result.Shape()

		if formPadding[i] > 0 {
			zeroshape[dim] = formPadding[i]

			zeros := G.NewTensor(g, input.Dtype(), dims, G.WithShape(zeroshape...), G.WithInit(G.Zeroes()))
			if i%2 == 0 {
				result, _ = G.Concat(dim, zeros, result)
			} else {
				result, _ = G.Concat(dim, result, zeros)
			}
		}

	}
	ss := make([]tensor.Slice, dims)
	shape := result.Shape()
	for j := 0; j < dims; j++ {
		ss[j] = G.S(0, shape[j])
	}

	for i := 0; i < 4; i++ {
		if formPadding[i] >= 0 {
			continue
		}
		dim := dims - 1 - i/2 // 维度
		if i%2 == 0 {
			ss[dim] = G.S(-1*formPadding[i], shape[dim])
		} else {
			ss[dim] = G.S(0, shape[dim]+formPadding[i])
		}
		shape[dim] = shape[dim] + formPadding[i]

	}
	result, _ = G.Slice(result, ss...)
	result = G.Must(G.Reshape(result, shape))

	return result, nil
}
