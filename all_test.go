package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"testing"
	"unet_gorgonia/utils"
)

func TestDataLoading(t *testing.T) {
	d := utils.NewDataset("./data/imgs", "./data/masks", "_mask.gif", 10)

	_, valSet := utils.RandomSplit(*d, 0.01)
	go utils.LoadImages(valSet, 2)
	a := <-utils.TrainChannel
	t.Log(a.Inputs.Data(), a.Inputs.Shape())
	t.Log(a.Masks.Data(), a.Masks.Shape())

}

func TestFile(t *testing.T) {
	f, err := os.Create("test.txt")
	defer f.Close()
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("world")
}

func TestCreateImage(t *testing.T) {
	d := utils.NewDataset("./data/imgs", "./data/masks", "_mask.gif", 10)

	_, valSet := utils.RandomSplit(*d, 0.01)
	go utils.LoadImages(valSet, 1)
	a := <-utils.TrainChannel

	//t.Log(a.Inputs.Data(), a.Inputs.Shape())
	//t.Log(a.Masks.Data(), a.Masks.Shape())

	//t.Log(reflect.TypeOf(a.Inputs.Data()))

	data := a.Inputs.Data().([]float64)
	dataMask := a.Masks.Data().([]float64)

	t.Log(len(data), a.Inputs.Shape())
	imageShape := a.Inputs.Shape()

	img := image.NewNRGBA64(image.Rect(0, 0, imageShape[3], imageShape[2]))
	imgMask := image.NewRGBA(image.Rect(0, 0, imageShape[3], imageShape[2]))
	for ii := 0; ii < imageShape[2]; ii++ {
		for jj := 0; jj < imageShape[3]; jj++ {

			//pix := projections[ii][jj] * 32767 / pixMaxt

			//projections[ii][jj] = projections[ii][jj] * 65535 / pixMaxt
			//
			//if projections[ii][jj] < 0 {
			//	log.Debug("llllllllllllll")
			//}
			////projections[ii][jj] = projections[ii][jj] / 10
			r := uint16(data[jj+ii*imageShape[3]] * 65535)
			g := uint16(data[jj+ii*imageShape[3]+imageShape[3]*imageShape[2]] * 65535)
			b := uint16(data[jj+ii*imageShape[3]+imageShape[3]*imageShape[2]*2] * 65535)
			a := uint16(65535)
			img.Set(jj, ii, color.RGBA64{r, g, b, a})

			r1 := uint8(dataMask[jj+ii*imageShape[3]] * 255)
			g2 := uint8(dataMask[jj+ii*imageShape[3]] * 255)
			b3 := uint8(dataMask[jj+ii*imageShape[3]] * 255)
			a2 := uint8(255)

			imgMask.Set(jj, ii, color.RGBA{r1, g2, b3, a2})
			//img.SetGray16(jj, ii, color.Gray16{uint16(projections[ii][jj])})
			//if imgMat.GetIntAt(jj, ii) < minNum {
			//	minNum = imgMat.GetIntAt(jj, ii)
			//}
		}
	}

	f, _ := os.Create(fmt.Sprintf("image%d.png", 1))
	defer f.Close()
	png.Encode(f, img)

	f2, _ := os.Create(fmt.Sprintf("mask%d.png", 2))
	defer f2.Close()
	png.Encode(f2, imgMask)
}
