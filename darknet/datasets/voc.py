#coding:utf-8

import torch.utils.data as data
from PIL import Image, ImageDraw
import os
import os.path
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

def default_loader(path):
    return Image.open(path).convert('RGB')

class TransformVOCDetectionAnnotation(object):
    def __init__(self, keep_difficult=False):
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj[0].text.lower().strip()
            bbox = obj[4]
            bndbox = [int(bb.text)-1 for bb in bbox]
            res += [bndbox + [name]]

        return res

class VOC(data.Dataset):
    def __init__(self, root, image_set, transform=None, target_transform=None, loader=default_loader):
        self.root = root
        self.image_set = image_set
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        dataset_name = 'VOC2012'
        self._annopath = os.path.join(self.root, dataset_name, 'Annotations', '%s.xml')
        self._imgpath = os.path.join(self.root, dataset_name, 'JPEGImages', '%s.jpg')
        self._imgsetpath = os.path.join(self.root, dataset_name, 'ImageSets', 'Main', '%s.txt')

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip('\n') for x in self.ids]

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()

        img = self.loader(os.path.join(self._imgpath % img_id))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def show(self, index):
        img, target = self.__getitem__(index)
        draw = ImageDraw.Draw(img)
        for obj in target:
            draw.rectangle(obj[0:4], outline=(255, 0, 0))
            draw.text(obj[0:2], obj[4], fill=(0, 255, 0))
        img.show()

if __name__ == '__main__':
    ds = VOC('G:/Python全套教程/VOCdevkit/', 'train', target_transform=TransformVOCDetectionAnnotation(False))
    print(len(ds))
    img, target = ds[0]
    print(target)
    ds.show(0)