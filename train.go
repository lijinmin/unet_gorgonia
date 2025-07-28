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
	input := G.NewTensor(g, dt, 4, G.WithName("input"))
	out, err := n.Forward(input)
	if err != nil {
		log.Fatal(err)
	}
	mask := G.NewTensor(g, dt, 4, G.WithName("mask"))

	//获取损失函数
	out2, err := G.SoftMax(out, 1)
	if err != nil {
		log.Fatal(err)
	}
	cost1, err := G.Mean(G.Must(G.Neg(G.Must(G.Sum(G.Must(G.HadamardProd(mask, G.Must(G.Log2(out2)))))))))
	if err != nil {
		log.Fatal(err)
	}

	if _, err = G.Grad(cost1, n.Learnables()...); err != nil {
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
