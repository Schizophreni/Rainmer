 
import cv2
import os
import random
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import copy
import glob


def prepare_Rain200H(datapath):
    input_path = os.path.join(datapath, 'rain/X2')
    target_path = os.path.join(datapath, 'norain')
    imgs = []
    gts = []
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain200H! total length: ", len(imgs))
    return imgs, gts

def prepare_Rain200L(datapath):
    input_path = os.path.join(datapath, 'rain')
    target_path = os.path.join(datapath, 'norain')
    imgs = []
    gts = []
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain200L! total length: ", len(imgs))
    return imgs, gts

def prepare_Rain800(datapath):
    input_path = os.path.join(datapath, 'rain')
    target_path = os.path.join(datapath, 'norain')
    imgs = []
    gts = []
    for i in range(700):
        target_file = "norain-%03d.png" % (i + 1)
        input_file = "rain-%03d.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain800! total length: ", len(imgs))
    return imgs, gts 

def prepare_Rain1400(datapath):
    input_path = os.path.join(datapath, 'rain')
    target_path = os.path.join(datapath, 'norain')
    imgs = []
    gts = []
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        for j in range(14):
            input_file = "%d_%d.jpg" % (i + 1, j + 1)
            imgs.append(os.path.join(input_path, input_file))
            gts.append(os.path.join(target_path, target_file))
    print("process DDN! total length: ", len(imgs))
    return imgs, gts

def prepare_Rain1200(datapath):
    # rainy and gt are connected together (left: rain, right gt)
    # imgs = []
    gts = []
    for i in range(12000):
        target_file = "%d.jpg" % (i+1)
        gts.append(os.path.join(datapath, target_file))
    print("process DID! total length: ", len(gts))
    return gts, gts

def prepare_spa(datapath):
    imgs, gts = [], []
    inputpath = os.path.join(datapath, "real_world")
    gtpath = os.path.join(datapath, "real_world_gt")
    num_scenes = len(os.listdir(inputpath))  # number of scenes
    for scene_idx in range(1, num_scenes+1):
        rain_path = os.path.join(inputpath, "{:03d}".format(scene_idx))
        gt_path = os.path.join(gtpath, "{:03d}".format(scene_idx))
        rscs = os.listdir(rain_path)  # rainy scenes
        for rsc in rscs:
            rsc_idx = rsc.split("-")[-1]
            rain_src_path = os.path.join(rain_path, "{:03d}-{}".format(scene_idx, rsc_idx))
            rain_imgs = os.listdir(rain_src_path)
            for rain_img in rain_imgs:
                x, y = rain_img.split(".")[0].split("_")[-2:]  # scene x, y coordinates
                imgs.append(os.path.join(rain_src_path, rain_img))
                gt_img = os.path.join(gt_path, "{:03d}_{}_{}.png".format(scene_idx, x, y))
                gts.append(gt_img)
    print("Total imgs: ", len(imgs), len(gts))
    return imgs, gts

# Synrain, gt-rain and gtav-balance
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
    valid_rains = []
    for rain_img in rains:
        if "0011_00_set2" in rain_img:
            print(rain_img)
            continue
        valid_rains.append(rain_img)
        rain_name = rain_img.split("/")[-1]
        gt_name = rain_name.split("_")
        gt_name = '_'.join(gt_name[:1] + gt_name[2:])
        gts.append(os.path.join(root_dir, mode, "gt", gt_name))
    print("GTAV total images: ", len(rains))
    return valid_rains, gts

class ContrastiveMixDataLoaderTrain(data.Dataset):
    def __init__(self, opt):
        super(ContrastiveMixDataLoaderTrain, self).__init__()
        self.opt = opt
        self.syn_imgs, self.syn_gts = [], []
        self.real_imgs, self.real_gts = [], []
        for data_path in opt.data_paths.split(","):
            data_path = data_path.strip()
            print(data_path)
            if data_path.find('Rain200H') != -1:
                imgs, gts = prepare_Rain200H(data_path)
            elif data_path.find('Rain200L') != -1:
                imgs, gts = prepare_Rain200L(data_path)
            elif data_path.find('Rain800') != -1:
                imgs, gts = prepare_Rain800(data_path)
            elif data_path.find('Rain1400') != -1:
                imgs, gts = prepare_Rain1400(data_path)
            elif data_path.find('Rain1200') != -1:
                imgs, gts = prepare_Rain1200(data_path)
            elif data_path.find("synrain") != -1:
                imgs, gts = prepare_synrain(dataset_name="synrain")
            elif data_path.find("gtrain") != -1:
                imgs, gts = prepare_gtrain(dataset_name="gt-rain")
            elif data_path.find("gtav-balance") != -1:
                imgs, gts = prepare_gtav_balance(dataset_name="gtav-balance")
            elif data_path.find('spa') != -1:
                imgs, gts = prepare_spa(data_path)
                self.real_imgs.extend(imgs)
                self.real_gts.extend(gts)
                continue
            else:
                raise (RuntimeError('Cannot find dataset!'))
            self.syn_imgs.extend(imgs)
            self.syn_gts.extend(gts)

        if len(self.syn_gts) == 0 and len(self.real_gts):
            raise(RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = self.syn_imgs
        self.gts = self.syn_gts
        # add real_world images
        if len(self.real_gts) > 0:
            print("Add real world images")
            self.add_real_world()
        self.crop_size = opt.crop_size
        # pre-load image        
        self.neg_transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=11, sigma=(0.3, 1.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        self.k_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.sizex = len(self.imgs)

        self.resize = transforms.Resize(size=(opt.crop_size, opt.crop_size), antialias=True)

    def __len__(self):
        tot_num = len(self.imgs)
        if tot_num % self.opt.batch_size == 0:
            return tot_num
        return (tot_num // self.opt.batch_size + 1) * self.opt.batch_size
    
    def add_real_world(self):
        if len(self.real_imgs) == 0:
            return
        print(len(self.real_imgs))
        del self.imgs, self.gts
        self.imgs, self.gts = copy.deepcopy(self.syn_imgs), copy.deepcopy(self.syn_gts)
        real_worlds = np.random.choice(range(len(self.real_gts)), len(self.syn_imgs), replace=False)
        for idx in real_worlds:
            self.imgs.append(self.real_imgs[idx])
            self.gts.append(self.real_gts[idx])

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        gt_path = self.gts[index_]
        # inp_img, tar_img = self.imgs_numpy[index_], self.gts_numpy[index_]
        inp_img, tar_img = cv2.imread(inp_path), cv2.imread(gt_path)
        inp_img, tar_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
        inp_img, tar_img = torch.from_numpy(inp_img).permute(2, 0, 1).contiguous(), torch.from_numpy(tar_img).permute(2, 0, 1).contiguous()
        if "Rain1200" in inp_path:
            inp_img, tar_img = torch.chunk(inp_img, chunks=2, dim=-1)
        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr = random.randint(0, hh - self.crop_size)
        cc = random.randint(0, ww - self.crop_size)
        # Crop patch
        inp_img = inp_img[:, rr:rr + self.crop_size, cc:cc + self.crop_size]
        tar_img = tar_img[:, rr:rr + self.crop_size, cc:cc + self.crop_size]
        # Crop patch Data Augmentations
        aug = random.randint(0, 3)
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        return inp_img, tar_img

def Rainloader(opt):
    dataset = ContrastiveMixDataLoaderTrain(opt)
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
    opt = argparse.ArgumentParser()
    opt = opt.parse_args()
    opt.data_paths = "/home/rw/Public/datasets/derain/Rain200H/train, /home/rw/Public/datasets/derain/Rain800/train"
    opt.img_size = 128
    opt.crop_size = 128
    opt.batch_size = 16
    dset = ContrastiveMixDataLoaderTrain(opt)
    """
    for _, data in enumerate(dset):
        r, g, resize_1, resize_2, _ = data
        r = r.permute(1, 2, 0)
        g = g.permute(1, 2, 0)
        resize_1, resize_2 = resize_1.permute(1, 2, 0), resize_2.permute(1, 2, 0)
        print(r.min(), r.max(), g.min(), g.max(), resize_1.min(), resize_1.max())
        
        plt.subplot(2, 2, 1)
        plt.imshow(r.numpy())
        plt.subplot(2, 2, 2)
        plt.imshow(g.numpy())
        plt.subplot(2, 2, 3)
        plt.imshow(resize_1.numpy())
        plt.subplot(2, 2, 4)
        plt.imshow(resize_2.numpy())
        plt.show()
        """