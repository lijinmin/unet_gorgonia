package utils

import (
	"encoding/json"
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
	ImagesDir   string
	MaskDir     string
	Scale       int
	MaskSuffix  string
	ImageSiffix string
	MaskValues  []string
}

func NewDataset(imagesDir, maskDir, maskSuffix string, scale int) *BasicDataset {
	if scale < 1 {
		log.Fatal("scale must be >= 1")
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
		ImagesDir:   imagesDir,
		MaskDir:     maskDir,
		MaskSuffix:  maskSuffix,
		Scale:       scale,
		IDs:         ids,
		ImageSiffix: imageSiffix,
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
		_ = filepath.Join(data.MaskDir, id+data.MaskSuffix)
	}

}
func (data *BasicDataset) loadImage(filename string) {

}
func (data *BasicDataset) preProcess(img image.Image, label string) (val tensor.Tensor) {
	rect := img.Bounds()

	if label == "mask" {
		channel1 := []float64{}
		channel2 := []float64{}
		for i := 0; i <= rect.Max.Y; i++ {
			if i%data.Scale != 0 {
				continue
			}
			for j := 0; j <= rect.Max.X; j++ {
				if j%data.Scale != 0 {
					continue
				}
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
		val = tensor.New(tensor.WithShape(1, 2, (rect.Max.Y/data.Scale+1), (rect.Max.X/data.Scale+1)), tensor.WithBacking(valData))

	} else {

		channel1 := []float64{}
		channel2 := []float64{}
		channel3 := []float64{}
		//ii, jj := 0, 0
		for i := 0; i <= rect.Max.Y; i++ {
			if i%data.Scale != 0 {
				continue
			}
			//ii += 1
			//jj = 0
			for j := 0; j <= rect.Max.X; j++ {
				if j%data.Scale != 0 {
					continue
				}
				//jj += 1
				//if jj == 192 {
				//	log.Debug(j, data.Scale)
				//}
				x1, x2, x3, a := img.At(j, i).RGBA()
				channel1 = append(channel1, float64(x1)/float64(a))
				channel2 = append(channel2, float64(x2)/float64(a))
				channel3 = append(channel3, float64(x3)/float64(a))
			}
		}
		//log.Debug(ii, jj)
		valData := append(channel1, channel2...)
		valData = append(valData, channel3...)
		val = tensor.New(tensor.WithShape(1, 3, (rect.Max.Y/data.Scale+1), (rect.Max.X/data.Scale+1)), tensor.WithBacking(valData))
	}
	return

}
func (data *BasicDataset) getitem(id string) (imageTensor tensor.Tensor, maskTensor tensor.Tensor) {
	var err error
	imageFile := path.Join(data.ImagesDir, id+data.ImageSiffix)
	maskFile := path.Join(data.MaskDir, id+data.MaskSuffix)
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

func (data *BasicDataset) SavetoFile(fileName string) {
	f, err := os.OpenFile(fileName, os.O_RDWR|os.O_CREATE, os.ModePerm)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	b, err := json.Marshal(data)
	if err != nil {
		log.Fatal(err)
	}
	f.Write(b)

}
func LoadFromFile(filename string) *BasicDataset {
	data := &BasicDataset{}
	b, err := os.ReadFile(filename)
	if err != nil {
		log.Fatal(err)
	}
	err = json.Unmarshal(b, data)
	if err != nil {
		log.Fatal(err)
	}
	return data

}
