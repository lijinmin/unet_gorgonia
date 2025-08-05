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
	"unet_gorgonia/utils"
)

func train(epochs int, n_channels, n_classes int, bilinear bool, bs int) {
	rand.Seed(1337)
	dt := tensor.Float64
	g := G.NewGraph()
	n := unet.NewUnet(g, n_channels, n_classes, false, dt)
	input := G.NewTensor(g, dt, 4, G.WithShape(bs, n_channels, 1280, 1918), G.WithName("input"))

	out, err := n.Forward(input)
	if err != nil {
		log.Fatal(err)
	}
	//mask := G.NewTensor(g, dt, 4, G.WithShape(bs, 2, 640, 959), G.WithName("mask"))
	mask := G.NewTensor(g, dt, 4, G.WithShape(bs, n_classes, 1280, 1918), G.WithName("mask"))

	//获取损失函数
	out2, err := G.SoftMax(out, 1)
	if err != nil {
		log.Fatal(err)
	}
	cost1 := G.Must(G.Neg(G.Must(G.Sum(G.Must(G.HadamardProd(mask, G.Must(G.Log2(out2)))), 1, 2, 3))))

	cost2 := diceLoss(out2, mask)
	log.Debug(cost1.Shape(), cost2.Shape())

	weight := 0.5
	alpha := G.NewConstant(weight)
	totalCost := G.Must(G.Mean(G.Must(G.Add(
		G.Must(G.Mul(alpha, cost1)),
		G.Must(G.Mul(G.NewConstant(1.0-weight), cost2)),
	))))
	//totalCost := G.Must(G.Mean(cost2))
	//log.Debug(totalCost.Shape())

	var costVal G.Value
	G.Read(totalCost, &costVal)

	//log.Debug(len(n.Learnables()))
	//totalCost = G.Must(G.Mean(out))
	if _, err = G.Grad(totalCost, n.Learnables()...); err != nil {
		log.Fatal(err)
	}

	prog, locMap, _ := G.Compile(g)
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(n.Learnables()...))
	solver := G.NewRMSPropSolver(G.WithBatchSize(float64(bs)))
	defer vm.Close()

	totalSet := utils.NewDataset("./data/imgs", "/data/masks", "_mask.gif", 1.0)
	trainSet, _ := utils.RandomSplit(*totalSet, 0.1)

	batches := len(trainSet.IDs) / bs
	log.Debugf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(80)
	for i := 0; i < epochs; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		utils.Shuffle(trainSet.IDs) // 打乱训练的图片顺序

		for b := 0; b < batches; b++ {
			var xVal, yVal tensor.Tensor
			a := <-utils.TrainChannel
			xVal = a.Inputs
			yVal = a.Masks
			G.Let(input, xVal)
			G.Let(mask, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", i, b, err)
			}
			solver.Step(G.NodesToValueGrads(n.Learnables()))
			vm.Reset()
			bar.Increment()
			if i%50 == 0 {
				log.Debug("current loss is ", costVal)
			}
		}
		time.Sleep(1 * time.Second)
	}
}

func diceLoss(input, target *G.Node) *G.Node {
	epsilon := G.NewConstant(1e-6)
	a := G.Must(G.Sum(G.Must(G.HadamardProd(input, target)), 1, 2, 3))
	b := G.Must(G.Add(G.Must(G.Sum(input, 1, 2, 3)), G.Must(G.Sum(target, 1, 2, 3))))
	//log.Debug(a.Shape(), b.Shape())
	//loss := G.Must(G.HadamardDiv(G.Must(G.Sub(G.Must(G.Add(b, epsilon)), G.Must(G.Add(a, epsilon)))), G.Must(G.Add(b, epsilon))))
	dice := G.Must(G.HadamardDiv(G.Must(G.Add(a, epsilon)), G.Must(G.Add(b, epsilon))))
	//log.Debug(loss.Shape())
	dice = G.Must(G.Mul(G.NewConstant(2.0), dice))

	loss := G.Must(G.Sub(G.NewConstant(1.0), dice))

	return loss
}
