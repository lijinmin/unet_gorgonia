package utils

import (
	"github.com/ngaut/log"
	"gorgonia.org/tensor"
	"image"
	_ "image/gif"  // 注册gif解码器
	_ "image/jpeg" // 注册jpeg解码器
	_ "image/png"  // 注册png解码器
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
)

type BasicDataset struct {
	IDs         []string
	imagesDir   string
	maskDir     string
	Scale       float64
	maskSuffix  string
	imageSiffix string
	maskValues  []string
}

func NewDataset(imagesDir, maskDir, maskSuffix string, scale float64) *BasicDataset {
	if scale <= 0 || scale > 1 {
		log.Fatal("scale must be between 0 and 1")
	}

	ids := []string{}
	imageSiffix := ""
	filepath.WalkDir(imagesDir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if f, err := os.Stat(p); err == nil && !f.IsDir() {
			_, s := path.Split(p)
			id := strings.TrimSpace(strings.Split(s, ".")[0])
			if len(id) != 0 {
				ids = append(ids, id)
				if imageSiffix == "" {
					imageSiffix = "." + strings.TrimSpace(strings.Split(s, ".")[1])
				}
			}

			ids = append(ids)
		}
		return nil
	})
	return &BasicDataset{
		imagesDir:   imagesDir,
		maskDir:     maskDir,
		maskSuffix:  maskSuffix,
		Scale:       scale,
		IDs:         ids,
		imageSiffix: imageSiffix,
	}
}

func loadImage(filename string) (image.Image, error) {
	file, err := os.Open(filename)
	defer file.Close()
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(file)
	return img, err
}

func (data *BasicDataset) Init() {

}

func (data *BasicDataset) uniqueMaskValues() {

	for _, id := range data.IDs {
		_ = filepath.Join(data.maskDir, id+data.maskSuffix)
	}

}
func (data *BasicDataset) loadImage(filename string) {

}
func (data *BasicDataset) preProcess(img image.Image, label string) (val tensor.Tensor) {
	rect := img.Bounds()

	if label == "mask" {
		channel1 := []float64{}
		channel2 := []float64{}
		for i := 0; i < rect.Max.Y; i++ {
			for j := 0; j < rect.Max.X; j++ {
				x1, _, _, _ := img.At(j, i).RGBA()
				if x1 == 0 {
					channel1 = append(channel1, 1)
					channel2 = append(channel2, 0)
				} else {
					channel1 = append(channel1, 0)
					channel2 = append(channel2, 1)
				}
			}
		}
		valData := append(channel1, channel2...)
		val = tensor.New(tensor.WithShape(1, 2, rect.Max.Y, rect.Max.X), tensor.WithBacking(valData))

	} else {

		channel1 := []float64{}
		channel2 := []float64{}
		channel3 := []float64{}
		for i := 0; i < rect.Max.Y; i++ {
			for j := 0; j < rect.Max.X; j++ {
				x1, x2, x3, a := img.At(j, i).RGBA()
				channel1 = append(channel1, float64(x1)/float64(a))
				channel2 = append(channel2, float64(x2)/float64(a))
				channel3 = append(channel3, float64(x3)/float64(a))
			}
		}
		valData := append(channel1, channel2...)
		valData = append(valData, channel3...)
		val = tensor.New(tensor.WithShape(1, 3, rect.Max.Y, rect.Max.X), tensor.WithBacking(valData))
	}
	return

}
func (data *BasicDataset) getitem(id string) (imageTensor tensor.Tensor, maskTensor tensor.Tensor) {
	var err error
	imageFile := path.Join(data.imagesDir, id+data.imageSiffix)
	maskFile := path.Join(data.maskDir, id+data.maskSuffix)
	img, err := loadImage(imageFile)
	if err != nil {
		log.Fatal(err)
	}
	imageTensor = data.preProcess(img, "image")

	img, err = loadImage(maskFile)
	if err != nil {
		log.Fatal(err)
	}
	maskTensor = data.preProcess(img, "mask")
	return
}
