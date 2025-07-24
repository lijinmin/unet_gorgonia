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
			bar.Increment()
		}
	}
}
