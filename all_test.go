package main

import (
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
