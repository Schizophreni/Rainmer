 
import cv2
import os
import random
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import glob
import copy


datasets = {
    "synrain": ("/home1/zhangsy/rh/data/derain/MultiRain/{}/rain", "/home1/zhangsy/rh/data/derain/MultiRain/{}/norain"),
    "gt-rain": ("/home1/zhangsy/rh/data/derain/GT-Rain/GT-RAIN_{}", ""),
    "gtav-balance": ("/home1/zhangsy/rh/data/derain/GTAV-balance/", "")
}

def prepare_synrain(dataset_name, mode="train"):
    rain_dir, norain_dir = datasets[dataset_name]
    rain_dir, norain_dir = rain_dir.format(mode), norain_dir.format(mode)
    rains = glob.glob(os.path.join(rain_dir, "*.jpg"))
    gts = []
    for rain in rains:
        rain_name = rain.split("/")[-1]
        gts.append(os.path.join(norain_dir, rain_name))
    print("total images: ", len(rains))
    return rains, gts

def prepare_gtrain(dataset_name, mode="train"):
    root_dir = datasets[dataset_name][0]
    root_dir = root_dir.format(mode)
    rains, gts = [], []
    scenes = os.listdir(root_dir)
    for scene in scenes:
        gt_image = glob.glob(os.path.join(root_dir, scene, "*C-000*.png"))
        rain_images = glob.glob(os.path.join(root_dir, scene, "*R-*.png"))
        gt_images = [gt_image[0] for _ in range(len(rain_images))]
        rains.extend(rain_images)
        gts.extend(gt_images)
        # print("scene: {}, total images: {}".format(scene, len(rain_images)))
    print("total images: ", len(rains))
    return rains, gts

def prepare_gtav_balance(dataset_name, mode="train"):
    root_dir = datasets[dataset_name][0]
    gts = []
    rains = glob.glob(os.path.join(root_dir, mode, "rainy/*.png"))
    for rain_img in rains:
        rain_name = rain_img.split("/")[-1]
        gt_name = rain_name.split("_")
        gt_name = '_'.join(gt_name[:1] + gt_name[2:])
        gts.append(os.path.join(root_dir, mode, "gt", gt_name))
    print("total images: ", len(rains))
    return rains, gts

class AnyRainDataset(data.Dataset):
    def __init__(self, opt):
        super(AnyRainDataset, self).__init__()
        self.opt = opt
        all_imgs, all_gts = [], []
        for dataset_name in opt.dataset_names.split(","):
            dataset_name = dataset_name.strip()
            if dataset_name.lower() == "synrain":
                imgs, gts = prepare_synrain(dataset_name, mode=opt.mode)
            elif dataset_name.lower() == "gt-rain":
                imgs, gts = prepare_gtrain(dataset_name, mode=opt.mode)
            elif dataset_name.lower() == "gtav-balance":
                imgs, gts = prepare_gtav_balance(dataset_name, mode=opt.mode)
            else:
                raise (RuntimeError('Cannot find dataset!'))
            all_imgs.append(imgs)
            all_gts.append(gts)

        if len(all_imgs) == 0:
            raise(RuntimeError("Found 0 images in ... "))
        self.imgs = [i for dset in all_imgs for i in dset]
        self.gts = [i for dset in all_gts for i in dset]
        self.sizex = len(self.imgs)
        self.hard_imgs, self.hard_gts = [], [] # hard imgs
        self.hard_index = 0  # current hard index 
        self.whole_index = 0 # current image index for whole evaluation
        self.crop_size = opt.crop_size
    
    def preprocess(self, inp_path, gt_path, resize=False):
        inp_img, gt_img = cv2.imread(inp_path), cv2.imread(gt_path)
        inp_img, gt_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        if resize:
            # resize inp and gt for fast evaluation
            inp_img = cv2.resize(inp_img, (self.crop_size, self.crop_size))
            gt_img = cv2.resize(gt_img, (self.crop_size, self.crop_size))
            inp_img, gt_img = torch.from_numpy(inp_img).permute(2, 0, 1).contiguous(), torch.from_numpy(gt_img).permute(2, 0, 1).contiguous()
            return inp_img, gt_img
        inp_img, gt_img = torch.from_numpy(inp_img).permute(2, 0, 1).contiguous(), torch.from_numpy(gt_img).permute(2, 0, 1).contiguous()
        hh, ww = gt_img.shape[1], gt_img.shape[2]
        rr = random.randint(0, hh - self.crop_size)
        cc = random.randint(0, ww - self.crop_size)
        # Crop patch
        inp_img = inp_img[:, rr:rr + self.crop_size, cc:cc + self.crop_size]
        gt_img = gt_img[:, rr:rr + self.crop_size, cc:cc + self.crop_size]
        # Crop patch Data Augmentations
        aug = random.randint(0, 3)
        if aug == 1:
            inp_img = inp_img.flip(1)
            gt_img = gt_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            gt_img = gt_img.flip(2)
        return inp_img, gt_img

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        gt_path = self.gts[index_]
        inp_img, gt_img = self.preprocess(inp_path, gt_path)
        if inp_path in self.hard_imgs:
            difficulty = 1 # hard
        else:
            difficulty = 0  # easy
        return inp_img.float() / 255.0, gt_img.float() / 255.0, difficulty
    
    def sample_hard(self, sample_num):
        # sample images only from hard 
        hard_num = len(self.hard_imgs)
        sample_inps, sample_gts = [], []
        for i in range(sample_num):
            inp_path, gt_path = self.hard_imgs[self.hard_index], self.hard_gts[self.hard_index]
            inp, gt = self.preprocess(inp_path, gt_path, resize=False)
            sample_inps.append(inp.unsqueeze(0))
            sample_gts.append(gt.unsqueeze(0))
            # update hard index
            self.hard_index = (1 + self.hard_index) % hard_num
        return torch.cat(sample_inps, dim=0).float() / 255.0, torch.cat(sample_gts, dim=0).float() / 255.0
    
    def sample_all(self, sample_num):
        # sample images from all data
        sample_inps, sample_gts = [], []
        inps_path, gts_path = [], []
        for i in range(sample_num):
            if self.whole_index >= self.sizex:
                break
            inp_path, gt_path = self.imgs[self.whole_index], self.gts[self.whole_index]
            inps_path.append(inp_path)
            gts_path.append(gt_path)
            inp, gt = self.preprocess(inp_path, gt_path, resize=True)
            sample_inps.append(inp.unsqueeze(0))
            sample_gts.append(gt.unsqueeze(0))
            # update hard index
            self.whole_index += 1
        return torch.cat(sample_inps, dim=0).float() / 255.0, torch.cat(sample_gts, dim=0).float() / 255.0, inps_path, gts_path
    
    def init(self, hard_imgs, hard_gts):
        self.hard_imgs = hard_imgs
        self.hard_gts = hard_gts
        self.hard_index, self.whole_index = 0, 0
        print("initialize dataset, Hard / All: [{}/{}]".format(len(hard_imgs), len(self.imgs)))
    
    def traverse(self, index):  # traverse images for validation or test
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        gt_path = self.gts[index_]
        print(index_, inp_path)
        if self.opt.preload:
            inp_img, gt_img = self.imgs_numpy[index_], self.gts_numpy[index_]
        else:
            inp_img, gt_img = cv2.imread(inp_path), cv2.imread(gt_path)
            inp_img, gt_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        inp_img, gt_img = torch.from_numpy(inp_img).permute(2, 0, 1).contiguous(), torch.from_numpy(gt_img).permute(2, 0, 1).contiguous()
        return inp_img, gt_img

def Anyrainloader(opt):
    dataset = AnyRainDataset(opt)
    print("Dataset Size:%d" %(len(dataset)))
    trainloader = data.DataLoader(dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True, drop_last=False)
    return trainloader

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    opt = argparse.ArgumentParser()
    opt = opt.parse_args()
    opt.dataset_names = "gt-rain, synrain, gtav-balance"
    opt.img_size = 128
    opt.crop_size = 128
    opt.batch_size = 16
    opt.preload = False
    opt.mode = "train"
    opt.num_workers = 0
    dset = Anyrainloader(opt)

    """
    for _, data in enumerate(dset):
        r, g, ds = data
        print(ds)
        r, g = r[0], g[0]
        r = r.permute(1, 2, 0) / 255.0
        g = g.permute(1, 2, 0) / 255.0
        print(r.min(), r.max(), g.min(), g.max())
        
        plt.subplot(1, 3, 1)
        plt.imshow(r.numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(g.numpy())
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(r.amax(dim=-1) - r.amin(dim=-1)).numpy())
        plt.show()
    """
    dset = AnyRainDataset(opt)
    inps, gts, inps_path, gts_path = dset.sample_all(sample_num=16)
    print(inps.shape, gts.shape)
    for i in range(inps.shape[0]):
        r, g = inps[i], gts[i]
        r = r.permute(1, 2, 0) / 255.0
        g = g.permute(1, 2, 0) / 255.0
        print(r.min(), r.max(), g.min(), g.max())
        print(inps_path[i], gts_path[i])
        
        plt.subplot(1, 3, 1)
        plt.imshow(r.numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(g.numpy())
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(r.amax(dim=-1) - r.amin(dim=-1)).numpy())
        plt.show()