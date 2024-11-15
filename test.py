"""
Test mix dataset, images are read with PIL
Test mix dataset, images are read with PIL
test models: BRN, RCDNet, DGUNet, IDT, DRSformer
test support: every model should contain a inference method for inference: [0, 1] to [0, 1]
special test: IDT: (patch_size 128x128, overlap: 32, merge with score_map)
"""

import cv2
import os
import numpy as np
import torch
from evaluation import rgb_to_y
from evaluation import psnr as compare_psnr
from evaluation import ssim as compare_ssim
import torch.nn.functional as F
import torchvision as TV
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from tqdm import tqdm
import glob
from utils.parse_config import parse
import importlib
import pdb
import shutil

opt = parse()

device = torch.device('cuda') if opt.train.use_GPU else torch.device("cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.train.gpu_id)


def obtain_crops(base_h, base_w, H, W, stride):
    stride_h, stride_w = stride, stride
    nh, nw = (H-base_h) // stride_h + 1, (W-base_w) // stride_w + 1
    hs, ws = [i*stride_h for i in range(nh)], [j*stride_w for j in range(nw)]
    if (H - base_h) % stride_h != 0:
        hs.append(H - base_h)
    if (W - base_w) % stride_w != 0:
        ws.append(W - base_w)
    return hs, ws

def obtain_score_map(base_h, base_w):
    ## obtain score map, ref: https://github.com/jiexiaou/IDT/blob/main/test_full_size.py
    hs, hw = torch.arange(base_h), torch.arange(base_w)
    hs = (hs - base_h / 2).unsqueeze(1)  # [H, 1]
    hw = (hw - base_w / 2).unsqueeze(0)  # [1, W]
    scores = 1.0 / torch.sqrt((hs**2 + hw**2 + 1e-3)).float()
    return scores[None, :, :]

# Test100, Rain100L, Rain100H, Test1200, Test2800 (Synthetic)
def test_test100():
    input_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Test100/input"
    target_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Test100/target"
    imgs, gts = [], []
    for i in range(98):
        input_file = "{}.png".format(i+1)
        target_file = "{}.png".format(i+1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("Prcess Test100 | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts

def test_rain100L():
    input_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Rain100L/input"
    target_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Rain100L/target"
    imgs, gts = [], []
    for i in range(100):
        input_file = "{}.png".format(i+1)
        target_file = "{}.png".format(i+1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("Prcess Rain100L | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts

def test_rain100H():
    input_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Rain100H/input"
    target_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Rain100H/target"
    imgs, gts = [], []
    for i in range(100):
        input_file = "{}.png".format(i+1)
        target_file = "{}.png".format(i+1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("Prcess Rain100H | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts

def test_Rain200H():
    input_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain200H/test/rain/X2')
    target_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain200H/test/norain')
    imgs = []
    gts = []
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain200H! total length: ", len(imgs))
    return imgs, gts

def test_test1200():
    input_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Test1200/input"
    target_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Test1200/target"
    imgs, gts = [], []
    for i in range(1200):
        input_file = "{}.png".format(i+1)
        target_file = "{}.png".format(i+1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("Prcess Test1200 | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts

def test_test2800():
    input_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Test2800/input"
    target_path = "/home1/zhangsy/rh/data/derain/MultiRain/test/Test2800/target"
    imgs, gts = [], []
    for i in range(800, 1000):
        for j in range(14):
            input_file = "{}_{}.jpg".format(i+1, j+1)
            target_file = "{}_{}.jpg".format(i+1, j+1)
            imgs.append(os.path.join(input_path, input_file))
            gts.append(os.path.join(target_path, target_file))
    print("Prcess Test2800 | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts

# Test GTAV-NightRain
def test_gtavfull(): # subset: set1 | set2 | set3 | hard
    imgs, gts = [], []
    input_path = os.path.join("/home1/zhangsy/rh/data/derain/gtavset3/test", "rainy")
    target_path = os.path.join("/home1/zhangsy/rh/data/derain/gtavset3/test", "gt")
    num_gt = len(os.listdir(target_path))
    for i in range(num_gt):
        for aug_idx in range(10):
            target_file = "{:04d}.png".format(i)
            input_file = "{:04d}_{:02d}.png".format(i, aug_idx)
            gts.append(os.path.join(target_path, target_file))
            imgs.append(os.path.join(input_path, input_file))
    print("Prcess GTAV-NightRain: {} | total files: [{}/{}]".format("set3", len(imgs), len(gts)))
    return imgs, gts

# test gtav crop
def test_gtavcrop():
    root_dir = "/home1/zhangsy/rh/data/derain/GTAV-balance/test"
    gts = []
    rains = glob.glob(os.path.join(root_dir, "rainy/*.png"))
    for rain_img in rains:
        rain_name = rain_img.split("/")[-1]
        gt_name = rain_name.split("_")
        gt_name = '_'.join(gt_name[:1] + gt_name[2:])
        gts.append(os.path.join(root_dir, "gt", gt_name))
    print("total images: ", len(rains))
    return rains, gts

# Test GT-Rain
def test_gtrain():
    root_dir = "/home1/zhangsy/rh/data/derain/GT-Rain/GT-RAIN_test"
    imgs, gts = [], []
    scenes = os.listdir(root_dir)
    for scene in scenes:
        gt_image = glob.glob(os.path.join(root_dir, scene, "*C-000*.png"))
        rain_images = glob.glob(os.path.join(root_dir, scene, "*R-*.png"))
        gt_images = [gt_image[0] for _ in range(len(rain_images))]
        imgs.extend(rain_images)
        gts.extend(gt_images)
        # print("scene: {}, total images: {}".format(scene, len(rain_images)))
    print("Prcess GT-Rain | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts

# Test Real_internet
def test_realint():
    input_path = "/home1/zhangsy/rh/data/derain/Real_Internet"
    imgs = glob.glob(os.path.join(input_path, "*.png"))
    print("Process Real Internet, total files: {}".format(len(imgs)))
    return imgs, imgs

# Test Real_ours
def test_realours():
    imgs = []
    input_path = "/home/wran/Public/datasets/derain/real-world/ours/"
    for i in range(1082):
        input_file = "{:03d}.jpg".format(i+1)
        imgs.append(os.path.join(input_path, input_file))
    print("Process real-world ours, total files: {}".format(len(imgs)))
    return imgs, imgs

# Test outdoor-rain
def test_outdoorrain():
    imgs, gts = [], []
    root_dir = "/home1/zhangsy/rh/data/derain/AllinOne/rain/test"
    rains = os.listdir(os.path.join(root_dir, "rain"))
    for rain_img in rains:
        imgs.append(os.path.join(root_dir, "rain", rain_img))
        norain_img = rain_img.split("_s")[0] + ".png"
        gts.append(os.path.join(root_dir, "gt", norain_img))
    print("Prcess Outdoor-Rain | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts

def test_raindrop():
    imgs, gts = [], []
    root_dir = "/home1/zhangsy/rh/data/derain/AllinOne/raindrop/test"
    for i in range(58):
        rain_img = f"{i}_rain.png"
        norain_img = f"{i}_clean.png"
        imgs.append(os.path.join(root_dir, "data", rain_img))
        gts.append(os.path.join(root_dir, "gt", norain_img))
    print("Prcess RainDrop | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts

def test_snowL():
    imgs, gts = [], []
    root_dir = "/home1/zhangsy/rh/data/derain/AllinOne/snow/test"
    snow_imgs = os.listdir(os.path.join(root_dir, "snowy"))
    for snow_img in snow_imgs:
        imgs.append(os.path.join(root_dir, "snowy", snow_img))
        gts.append(os.path.join(root_dir, "gt", snow_img))
    print("Prcess Snow100K-L | total files: [{}/{}]".format(len(imgs), len(gts)))
    return imgs, gts


def obtain_test(dataset_name):
    if dataset_name == "test100":
        imgs, gts = test_test100()
    elif dataset_name == "rain100L":
        imgs, gts = test_rain100L()
    elif dataset_name == "rain100H":
        imgs, gts = test_rain100H()
    elif dataset_name == "test1200":
        imgs, gts = test_test1200()
    elif dataset_name == "test2800":
        imgs, gts = test_test2800()
    elif dataset_name == "gtrain":
        imgs, gts = test_gtrain()
    elif "gtavset3" in dataset_name:
        imgs, gts = test_gtavfull()
    elif "gtavcrop" in dataset_name:
        imgs, gts = test_gtavcrop()
    elif dataset_name == "realint":
        imgs, gts = test_realint()
    elif dataset_name == "real_ours":
        imgs, gts = test_realours()
    elif dataset_name == "outdoorrain":
        imgs, gts = test_outdoorrain()
    elif dataset_name == "raindrop":
        imgs, gts = test_raindrop()
    elif dataset_name == "snow100k-L":
        imgs, gts = test_snowL()
    elif dataset_name == "rain200H":
        imgs, gts = test_Rain200H()
    return imgs, gts
    
def obtain_model(opt):
    # define model
    model = importlib.import_module(opt.model.model.split("-")[0].strip())  # import module
    model = getattr(model, opt.model.model.split("-")[-1].strip())(opt.model)  # instantiate model
    model.to(device)
    print("[===] Build Model ...")
    model.load_state_dict(torch.load(opt.checkpoint), strict=True)
    print("[===] Load checkpoint finished ...")
    model.eval()
    return model

def merge_img(base_h, base_w, hs, ws, all_crops, H, W, use_score_map=False):
    mask = torch.zeros(3, H, W)
    out = torch.zeros(3, H, W)
    all_crops = torch.cat(all_crops, dim=0)
    if use_score_map:
        score_map = obtain_score_map(base_h, base_w)
    else:
        score_map = 1
    cnt = 0
    for h in hs:
        for w in ws:
            out[:, h:h+base_h, w:w+base_w] += all_crops[cnt] * score_map
            mask[:, h:h+base_h, w:w+base_w] += score_map
            cnt += 1
    return out / mask

@torch.no_grad()
def main():
    os.makedirs(os.path.join(opt.save_path), exist_ok=True)
    psnrs, ssims = [], []
    cnt = 0
    model = obtain_model(opt)
    # parse images
    imgs, gts = obtain_test(dataset_name=opt.testset)
    save_img = True
    with tqdm(zip(imgs, gts), ncols=100) as pbar_test:
        for rain_img, gt_img in pbar_test:
            # print(rain_img, gt_img)
            rain_file = rain_img
            rain_name = rain_img.split("/")[-1]
            inp_img, gt_img = cv2.imread(rain_img), cv2.imread(gt_img)
            if "set3" in opt.testset:
                H, W = inp_img.shape[:2]
                inp_img = cv2.resize(inp_img, (int(W/H*512), 512))
                gt_img = cv2.resize(gt_img, (int(W/H*512), 512))
            
            inp_img, gt_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            resize_inp = cv2.resize(inp_img, (128, 128))
            inp_img, gt_img = torch.from_numpy(inp_img).permute(2, 0, 1).contiguous(), torch.from_numpy(gt_img).permute(2, 0, 1).contiguous()
            x, y = inp_img.to(device, non_blocking=True).float() / 255.0, gt_img.float() / 255.0
            resize_inp = torch.from_numpy(resize_inp).permute(2, 0, 1).contiguous()
            resize_inp = resize_inp.unsqueeze(0).to(device, non_blocking=True).float() / 255.0
            
            H, W = x.shape[-2], x.shape[-1]
            
            if min(H, W) <= opt.tile:
                if opt.model_name in ["DGUNet", "DRSformer"]:
                    pad_h, pad_w = 8 - H % 8 if H % 8 !=0 else 0, 8 - W % 8 if W % 8 != 0 else 0
                    x = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), "reflect").to(device)
                elif opt.model_name == "IDT":
                    pad_h, pad_w = 8 - H % 8 if H % 8 !=0 else 0, 8 - W % 8 if W % 8 != 0 else 0
                    if H < opt.tile:
                        pad_h = opt.tile - H
                    if W < opt.tile:
                        pad_w = opt.tile - W
                    x = F.pad(x.unsqueeze(0), (0, pad_w, 0, pad_h), "reflect").to(device)
                else:
                    x = x.unsqueeze(0).to(device)
                # use inference function 
                # print(x.shape)
                out = model.inference(x, im_q=resize_inp).cpu()
                out_img = out[0, :, :H, :W].clamp(0.0, 1.0)
            else:
                base_h, base_w = opt.tile, opt.tile
                hs, ws = obtain_crops(base_h=base_h, base_w=base_w, H=H, W=W, stride=opt.tile-opt.tile_overlap)
                batch_cnt = 0  
                x_batch = []
                all_crops = []
                batch_size =  (512 // opt.tile)**2
                for start_h in hs:
                    for start_w in ws:
                        batch_cnt += 1
                        crop_x = x[:, start_h:start_h+base_h, start_w:start_w+base_w].unsqueeze(0)
                        x_batch.append(crop_x)
                        cnt += 1
                        if batch_cnt == batch_size or (start_h == hs[-1] and start_w == ws[-1]):
                            crop_xs = torch.cat(x_batch, dim=0).to(device)
                            outs = model.inference(crop_xs, im_q=resize_inp)
                            outs.clamp_(0.0, 1.0)
                            x_batch = []
                            batch_cnt = 0
                            if save_img:
                                all_crops.append(outs.cpu())
                if opt.model_name == "IDT":
                    out_img = merge_img(base_h, base_w, hs, ws, all_crops, H, W, use_score_map=True)
                else:
                    out_img = merge_img(base_h, base_w, hs, ws, all_crops, H, W, use_score_map=False)
                out_img.clamp_(0.0, 1.0)
            x = x.cpu()
            # print(x.min(), x.max(), ((x-y)**2).mean())
            # pdb.set_trace()
            psnr_val = compare_psnr(rgb_to_y(out_img.cpu()*255.0), rgb_to_y(y*255.0))
            ssim_val = compare_ssim(rgb_to_y(out_img.cpu()*255.0), rgb_to_y(y*255.0))
            # print(out_img.permute(1, 2, 0).cpu().numpy().shape)
            pbar_test.set_postfix(psnr=psnr_val, ssim=ssim_val, H=H, W=W)
            psnrs.append(psnr_val)
            ssims.append(ssim_val)

            if save_img:
                TV.utils.save_image(out_img, os.path.join(opt.save_path, rain_name))
                # TV.utils.save_image(y, os.path.join(opt.save_path, "GT-"+rain_name))
    
    print(np.array(psnrs).shape)
    print('Avg. psnr: ', np.array(psnrs).mean())
    print('Avg. ssim: ', np.array(ssims).mean())
    print('Total patches: ', cnt)

if __name__ == "__main__":
    main()
    a = input("hello...")  # avoid using screen testing directly exit
