package utils

import "log"

type BasicDataset struct {
	IDs        []string
	imagesDir  string
	maskDir    string
	Scale      float64
	maskSuffix string
}

func (data *BasicDataset) NewDataset(imagesDir, maskDir, maskSuffix string, scale float64) *BasicDataset {
	if scale <= 0 || scale > 1 {
		log.Fatal("scale must be between 0 and 1")
	}
	return &BasicDataset{
		imagesDir:  imagesDir,
		maskDir:    maskDir,
		maskSuffix: maskSuffix,
		Scale:      scale,
	}
}

func (data *BasicDataset) Init() {

}
func (data *BasicDataset) loadImage() {

}
func (data *BasicDataset) preProcess() {

}
func (data *BasicDataset) getitem(id string) {

}
