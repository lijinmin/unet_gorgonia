package main

import (
	"fmt"
	"github.com/ngaut/log"
	pb "gopkg.in/cheggaaa/pb.v1"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"math/rand"
	"time"
	"unet_gorgonia/unet"
)

func train(epochs int, n_channels, n_classes int, bilinear bool) {
	numExamples := 1000
	rand.Seed(1337)
	dt := tensor.Float64
	bs := 4
	g := G.NewGraph()
	n := unet.NewUnet(g, n_channels, n_classes, false, dt)
	input := G.NewTensor(g, dt, 4, G.WithShape(bs, n_channels, 640, 959), G.WithName("input"))

	out, err := n.Forward(input)
	if err != nil {
		log.Fatal(err)
	}
	//mask := G.NewTensor(g, dt, 4, G.WithShape(bs, 2, 640, 959), G.WithName("mask"))
	mask := G.NewTensor(g, dt, 4, G.WithShape(bs, n_classes, 80, 119), G.WithName("mask"))

	//获取损失函数
	out2, err := G.SoftMax(out, 1)
	if err != nil {
		log.Fatal(err)
	}
	cost1, err := G.Mean(G.Must(G.Neg(G.Must(G.HadamardProd(mask, G.Must(G.Log2(out2)))))))
	if err != nil {
		log.Fatal(err)
	}
	cost2 := diceLoss(g, dt, out2, mask)

	weight := 0.5
	alpha := G.NewConstant(weight)
	totalCost := G.Must(G.Add(
		G.Must(G.Mul(alpha, cost1)),
		G.Must(G.Mul(G.NewConstant(1.0-weight), cost2)),
	))

	var costVal G.Value
	G.Read(totalCost, &costVal)

	log.Debug(len(n.Learnables()))
	totalCost = G.Must(G.Mean(out))
	if _, err = G.Grad(totalCost, n.Learnables()...); err != nil {
		log.Fatal(err)
	}

	prog, locMap, _ := G.Compile(g)
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(n.Learnables()...))
	solver := G.NewRMSPropSolver(G.WithBatchSize(float64(bs)))
	defer vm.Close()

	batches := numExamples / bs
	log.Debugf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)
	for i := 0; i < epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()

		for i := 0; i < batches; i++ {
			solver.Step(G.NodesToValueGrads(n.Learnables()))
			vm.Reset()
			bar.Increment()
		}
	}
}

func diceLoss(g *G.ExprGraph, dt tensor.Dtype, input, target *G.Node) *G.Node {
	size := input.Shape()[0]
	epsilon := G.NewMatrix(g, dt, G.WithShape(1, size), G.WithName("epsilon"), G.WithInit(G.ValuesOf(1e-6)))
	a := G.Must(G.Sum(G.Must(G.HadamardProd(input, target)), 1, 2, 3))
	b := G.Must(G.Add(G.Must(G.Sum(input, 1, 2, 3)), G.Must(G.Sum(target, 1, 2, 3))))
	dice := G.Must(G.Mean(G.Must(G.HadamardDiv(G.Must(G.Add(a, epsilon)), G.Must(G.Add(b, epsilon))))))

	loss := G.Must(G.Sub(G.NewConstant(1.0), dice))
	return loss
}
