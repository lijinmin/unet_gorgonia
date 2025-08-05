package main

import (
	"testing"
	"unet_gorgonia/utils"
)

func TestDataLoading(t *testing.T) {
	d := utils.NewDataset("./data/imgs", "/data/masks", "_mask.gif", 1.0)

	trainSet, valSet := utils.RandomSplit(*d, 0.1)
	t.Log(len(d.IDs), len(trainSet.IDs), valSet.IDs)
	utils.Shuffle(valSet.IDs)
	t.Log(valSet.IDs)
}
