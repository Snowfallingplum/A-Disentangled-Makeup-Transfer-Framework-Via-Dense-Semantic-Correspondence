import os
import torch
import numpy as np
from PIL import Image
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
class MakeupDataset(data.Dataset):
    def __init__(self,opts):
        self.opt=opts
        self.dataroot=opts.dataroot
        # non_makeup
        name_non_makeup = os.listdir(os.path.join(self.dataroot, 'non-makeup'))
        self.non_makeup_path = [os.path.join(self.dataroot, 'non-makeup', x) for x in name_non_makeup]
        # makeup
        name_makeup = os.listdir(os.path.join(self.dataroot, 'makeup'))
        self.makeup_path = [os.path.join(self.dataroot, 'makeup', x) for x in name_makeup]

        self.warproot=os.path.join(self.dataroot, 'warp')

        self.non_makeup_size = len(self.non_makeup_path)
        self.makeup_size = len(self.makeup_path)
        # self.dataset_size = max(self.non_makeup_size, self.makeup_size)
        if self.opt.phase=='train':
            self.dataset_size = self.non_makeup_size
        else:
            self.dataset_size = self.non_makeup_size*self.makeup_size

        self.image_transform = transforms.Compose([transforms.Resize((320, 320), Image.BILINEAR),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def load_img(self, img_name):
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        if self.opt.phase=='test':
            non_makeup_index=index//self.makeup_size
            makeup_index=index%self.makeup_size
            print(self.non_makeup_size,self.makeup_size,non_makeup_index,makeup_index)
            non_makeup_img = self.load_img(self.non_makeup_path[non_makeup_index])
            makeup_img = self.load_img(self.makeup_path[makeup_index])
            data_pre=self.test_preprocessing(self.opt,non_makeup_img=non_makeup_img,makeup_img=makeup_img)
            non_makeup_img = data_pre['non_makeup']
            makeup_img = data_pre['makeup']
            non_makeup_norm = (non_makeup_img + 1.) / 2. * 255.
            non_makeup_norm = Image.fromarray(non_makeup_norm.astype('uint8'))
            makeup_norm = (makeup_img + 1.) / 2. * 255.
            makeup_norm = Image.fromarray(makeup_norm.astype('uint8'))

            non_makeup_img = np.transpose(non_makeup_img, (2, 0, 1))
            makeup_img = np.transpose(makeup_img, (2, 0, 1))
            data = {'non_makeup': torch.from_numpy(non_makeup_img).type(torch.FloatTensor),
                    'makeup': torch.from_numpy(makeup_img).type(torch.FloatTensor),
                    'non_makeup_norm': self.image_transform(non_makeup_norm),
                    'makeup_norm': self.image_transform(makeup_norm)}
            return data

    def test_preprocessing(self,opts,non_makeup_img,makeup_img):
        non_makeup_img=cv2.resize(non_makeup_img,(opts.resize_size,opts.resize_size))
        makeup_img = cv2.resize(makeup_img, (opts.resize_size, opts.resize_size))
        h1 = int((opts.resize_size - opts.crop_size) / 2)
        w1 = int((opts.resize_size - opts.crop_size) / 2)
        non_makeup_img = non_makeup_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
        makeup_img = makeup_img[h1:h1 + opts.crop_size, w1:w1 + opts.crop_size]
        non_makeup_img = non_makeup_img / 127.5 - 1.
        makeup_img = makeup_img / 127.5 - 1.
        data = {'non_makeup': non_makeup_img, 'makeup': makeup_img}
        return data

    def __len__(self):
        return self.dataset_size



