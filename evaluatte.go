package main

import (
	"fmt"
	"github.com/ngaut/log"
	"gopkg.in/cheggaaa/pb.v1"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
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
	n := unet.NewUnet(g, n_channels, n_classes, false, dt) //
	input := G.NewTensor(g, dt, 4, G.WithShape(bs, n_channels, int(1280/scale+1), int(1918/scale+1)), G.WithName("input"))

	out, err := n.Forward(input)
	if err != nil {
		log.Fatal(err)
	}

	outMax := G.Must(G.Reshape(G.Must(G.Max(out, 1)), tensor.Shape{bs, 1, int(1280/scale + 1), int(1918/scale + 1)}))
	out1 := G.Must(G.Sub(out, G.Must(G.Concat(1, outMax, outMax))))

	//G.LeakyRelu()
	preMask := G.Must(G.Rectify(out1))
	var preMaskVal G.Value
	G.Read(preMask, &preMaskVal)

	////mask := G.NewTensor(g, dt, 4, G.WithShape(bs, 2, 640, 959), G.WithName("mask"))

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
	bar.Prefix(fmt.Sprintf("Epoch %d", 1))
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
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", 1, b, err)
		}
		saveImgs(xVal, yVal, preMaskVal, b)
		vm.Reset()
		bar.Increment()

		break

	}
}

func saveImgs(img, mask tensor.Tensor, preMask G.Value, index int) {
	log.Debug(mask.Shape(), preMask.Shape())
	log.Debug(preMask.Data())
	data := img.Data().([]float64)
	dataMask := mask.Data().([]float64)

	imageShape := mask.Shape()

	oriImg := image.NewNRGBA64(image.Rect(0, 0, imageShape[3], imageShape[2]))
	imgMask := image.NewRGBA(image.Rect(0, 0, imageShape[3], imageShape[2]))
	for ii := 0; ii < imageShape[2]; ii++ {
		for jj := 0; jj < imageShape[3]; jj++ {

			r := uint16(data[jj+ii*imageShape[3]] * 65535)
			g := uint16(data[jj+ii*imageShape[3]+imageShape[3]*imageShape[2]] * 65535)
			b := uint16(data[jj+ii*imageShape[3]+imageShape[3]*imageShape[2]*2] * 65535)
			a := uint16(65535)
			oriImg.Set(jj, ii, color.RGBA64{r, g, b, a})

			r1 := uint8(dataMask[jj+ii*imageShape[3]] * 255)
			g2 := uint8(dataMask[jj+ii*imageShape[3]] * 255)
			b3 := uint8(dataMask[jj+ii*imageShape[3]] * 255)
			a2 := uint8(255)

			imgMask.Set(jj, ii, color.RGBA{r1, g2, b3, a2})

		}
	}

	f, _ := os.Create(fmt.Sprintf("./evaluation/image%d.png", index))
	defer f.Close()
	png.Encode(f, oriImg)

	f2, _ := os.Create(fmt.Sprintf("./evaluation/mask%d.png", index))
	defer f2.Close()
	png.Encode(f2, imgMask)
}
