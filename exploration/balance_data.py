import numpy as np
import os
from parse_data import parse_dataset, datasets
import shutil
import cv2
from tqdm import tqdm


def calc_densities(dataset_name, threshold=8 / 255.0):  # rain map
    rains, gts = parse_dataset(dataset_name)
    densities = []
    # print(rains, gts)
    """
    for rain_img, gt_img in tqdm(zip(rains, gts), ncols=80):
        # calculate rain map
        rain, gt = cv2.imread(rain_img) / 255.0, cv2.imread(gt_img) / 255.0
        diff = np.abs(rain - gt).max(axis=-1) 
        rho = diff / (rain.max(axis=-1) + gt.max(axis=-1) + 1e-3)
        rho = rho.mean()
        densities.append(rho)
        rains.append(rain_img)
        gts.append(gt_img)
    """
    # densities = np.array(densities, dtype=np.float32)
    return densities, rains, gts


# balance gtav-nightrain dataset according to GT-Rain and SynRain (only 0.00-0.08 should be considered)
syn_densities = np.load("SynRain_density.npy")
gtrain_densities = np.load("GT-Rain_density.npy")

syn_rho_004 = (syn_densities <= 0.04).sum() / syn_densities.shape[0]
gtrain_rho_004 = (gtrain_densities <= 0.04).sum() / gtrain_densities.shape[0]

rho_004 = 0.5 * (syn_rho_004 + gtrain_rho_004)

syn_rho_004_008 = ((syn_densities > 0.04) * (syn_densities <= 0.08)).sum() / syn_densities.shape[0]
gtrain_rho_004_008 = ((gtrain_densities > 0.04) * (gtrain_densities <= 0.08)).sum() / gtrain_densities.shape[0]

rho_004_008 = 0.5 * (syn_rho_004_008 + gtrain_rho_004_008)

print(syn_rho_004, gtrain_rho_004, syn_rho_004_008, gtrain_rho_004_008)

dataset_name = "GTAV-balance"
save_dir = datasets[dataset_name][0]
print(save_dir)

densities, rains, gts = calc_densities("GTAV-crop")
densities = np.load("GTAV-crop_density.npy")

num_larger_008 = (densities > 0.08).sum()
rho_larger_008 = 1 - rho_004 - rho_004_008
tot_num = num_larger_008 / rho_larger_008

num_004 = int(tot_num * rho_004)
num_004_008 = int(tot_num * rho_004_008)
print(num_004, num_004_008, tot_num)
indices = np.arange(densities.shape[0])
indices_004 = indices[(densities <= 0.04)]
select_004 = np.random.choice(indices_004, num_004, replace=False)
indices_004_008 = indices[(densities > 0.04) * (densities <= 0.08)]
select_004_008 = np.random.choice(indices_004_008, num_004_008, replace=False)

indices_larger_008 = indices[densities > 0.08]

print(len(rains), len(gts))
"""

for idx in select_004:
    rain_img, gt_img = rains[idx], gts[idx]
    dst_rain = os.path.join(save_dir, "train/rainy", rain_img.split("/")[-1])
    dst_gt = os.path.join(save_dir, "train/gt", gt_img.split("/")[-1])
    shutil.copy(rain_img, dst_rain)
    shutil.copy(gt_img, dst_gt)

for idx in select_004_008:
    rain_img, gt_img = rains[idx], gts[idx]
    dst_rain = os.path.join(save_dir, "train/rainy", rain_img.split("/")[-1])
    dst_gt = os.path.join(save_dir, "train/gt", gt_img.split("/")[-1])
    shutil.copy(rain_img, dst_rain)
    shutil.copy(gt_img, dst_gt)
"""

for idx in indices_larger_008:
    rain_img, gt_img = rains[idx], gts[idx]
    dst_rain = os.path.join(save_dir, "train/rainy", rain_img.split("/")[-1])
    dst_gt = os.path.join(save_dir, "train/gt", gt_img.split("/")[-1])
    shutil.copy(rain_img, dst_rain)
    shutil.copy(gt_img, dst_gt)



