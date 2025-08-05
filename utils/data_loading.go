package utils

import (
	"github.com/ngaut/log"
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
	IDs        []string
	imagesDir  string
	maskDir    string
	Scale      float64
	maskSuffix string
	maskValues []string
}

func NewDataset(imagesDir, maskDir, maskSuffix string, scale float64) *BasicDataset {
	if scale <= 0 || scale > 1 {
		log.Fatal("scale must be between 0 and 1")
	}

	ids := []string{}
	filepath.WalkDir(imagesDir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if f, err := os.Stat(p); err == nil && !f.IsDir() {
			_, s := path.Split(p)
			id := strings.TrimSpace(strings.Split(s, ".")[0])
			if len(id) != 0 {
				ids = append(ids, id)
			}

			ids = append(ids)
		}
		return nil
	})
	return &BasicDataset{
		imagesDir:  imagesDir,
		maskDir:    maskDir,
		maskSuffix: maskSuffix,
		Scale:      scale,
		IDs:        ids,
	}
}

func loadImage(s string) (image.Image, error) {
	file, err := os.Open(s)
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
func (data *BasicDataset) preProcess() {

}
func (data *BasicDataset) getitem(id string) {

}
