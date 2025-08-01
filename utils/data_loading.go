package utils

type BasicDataset struct {
	IDs        []string
	imagesDir  string
	maskDir    string
	Scale      float64
	maskSuffix string
}

func (data *BasicDataset) NewDataset() *BasicDataset {
	return &BasicDataset{}
}

func (data *BasicDataset) Init() {

}
func (data *BasicDataset) loadImage() {

}
func (data *BasicDataset) preProcess() {

}
func (data *BasicDataset) getitem(id string) {

}
