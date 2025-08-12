package main

import (
	"fmt"
	"github.com/ngaut/log"
	"gopkg.in/cheggaaa/pb.v1"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"math/rand"
	"time"
	"unet_gorgonia/unet"
	"unet_gorgonia/utils"
)

func evaluate(n_channels, n_classes int, bs int) {
	G.EvalMode()
	scale := 10
	rand.Seed(1337)
	dt := tensor.Float64
	g := G.NewGraph()
	n := unet.NewUnet(g, n_channels, n_classes, false, dt)
	input := G.NewTensor(g, dt, 4, G.WithShape(bs, n_channels, int(1280/scale+1), int(1918/scale+1)), G.WithName("input"))

	out, err := n.Forward(input)
	if err != nil {
		log.Fatal(err)
	}

	////mask := G.NewTensor(g, dt, 4, G.WithShape(bs, 2, 640, 959), G.WithName("mask"))
	mask := G.NewTensor(g, dt, 4, G.WithShape(bs, n_classes, int(1280/scale+1), int(1918/scale+1)), G.WithName("mask"))

	prog, locMap, _ := G.Compile(g)
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(n.Learnables()...))
	defer vm.Close()

	totalSet := utils.NewDataset("./data/imgs", "./data/masks", "_mask.gif", scale)
	_, evalSet := utils.RandomSplit(*totalSet, 0.1)

	batches := len(evalSet.IDs) / bs
	log.Debugf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)
	G.EvalMode()
	bar.Prefix(fmt.Sprintf("Epoch %d", i))
	bar.Set(0)
	bar.Start()
	utils.Shuffle(evalSet.IDs) // 打乱训练的图片顺序

	go utils.LoadImages(evalSet, bs)

	for b := 0; b < batches; b++ {
		var xVal, yVal tensor.Tensor
		a := <-utils.TrainChannel
		//log.Debug(a.Inputs.Shape(), a.Masks.Shape())
		xVal = a.Inputs
		yVal = a.Masks
		G.Let(input, xVal)
		G.Let(mask, yVal)
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", i, b, err)
		}
		vm.Reset()
		bar.Increment()

	}
}
