package utils

import (
	"fmt"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"math/rand"
	"time"
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

func Shuffle(slice []string) {
	// 使用当前时间初始化随机数生成器
	rand.Seed(time.Now().UnixNano())
	n := len(slice)
	for i := 0; i < n; i++ {
		// 生成一个[i, n)范围内的随机索引
		j := i + rand.Intn(n-i)
		// 交换元素
		slice[i], slice[j] = slice[j], slice[i]
	}
}

func RandomSplit(dataSet BasicDataset, valPercent float64) (trainSet *BasicDataset, valSet *BasicDataset) {
	trainSet = &BasicDataset{
		imagesDir:  dataSet.imagesDir,
		maskSuffix: dataSet.maskSuffix,
		maskDir:    dataSet.maskDir,
		Scale:      dataSet.Scale,
	}
	valSet = &BasicDataset{
		imagesDir:  dataSet.imagesDir,
		maskSuffix: dataSet.maskSuffix,
		maskDir:    dataSet.maskDir,
		Scale:      dataSet.Scale,
	}

	valSetLenght := int(float64(len(dataSet.IDs)) * valPercent)
	valIds := []string{}
	trainIds := []string{}
	for i, l := range dataSet.IDs {
		if i < valSetLenght {
			valIds = append(valIds, l)
		} else {
			trainIds = append(trainIds, l)
		}

	}
	trainSet.IDs = trainIds
	valSet.IDs = valIds

	return
}

type TrainData struct {
	Inputs tensor.Tensor
	Masks  tensor.Tensor
}

var (
	TrainChannel = make(chan TrainData, 100)
)

func LoadImages(ids []string, bs int) {

}
