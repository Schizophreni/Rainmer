import cv2
import numpy as np
from parse_data import parse_dataset
from tqdm import tqdm
import os
import torch

# compare different datasets from seven aspects:
# 1. total entropy of gt
# 2. total entropy of rain
# 3. psnr interval of (gt, rain)
# 4. density of rain interval
# 5. y gradients interval for rain
# 6. illuminance interval
# 7. monochromatic of rain (std(r, g, b))


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)

def psnr(x, y, data_range=255.0):
    x, y = x / data_range, y / data_range
    mse = torch.mean((x - y) ** 2)
    score = - 10 * torch.log10(mse)
    return score

def entropy(image: np.array, eps=1e-5):
    # image: [H, W, C]
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    r_bins, _ = np.histogram(r, bins=16, range=(0, 255))
    g_bins, _ = np.histogram(g, bins=16, range=(0, 255))
    b_bins, _ = np.histogram(b, bins=16, range=(0, 255))
    r_bins, g_bins, b_bins = r_bins + eps, g_bins + eps, b_bins + eps
    r_prob, g_prob, b_prob = r_bins / r_bins.sum(), g_bins / g_bins.sum(), b_bins / b_bins.sum()
    r_entropy = - (r_prob * np.log2(r_prob)).sum()
    g_entropy = - (g_prob * np.log2(g_prob)).sum()
    b_entropy = - (b_prob * np.log2(b_prob)).sum()
    return (r_entropy + g_entropy + b_entropy) / 3.0

def cal_y_gradients(image: np.array, mask):
    grad = image[1:, :, :] - image[:-1, :, :]
    grad = grad[mask[1:,:]]
    if len(grad) == 0:
        return None
    else:
        return np.abs(grad).mean()

def cal_density(rain, gt):
    diff = np.abs(rain - gt).max(axis=-1) 
    rho = diff / (rain.max(axis=-1) + gt.max(axis=-1) + 1e-3)
    rho = rho.mean()
    return rho

def illuminance(image: np.array):
    return image.mean()

def non_monochromatic(image: np.array, mask):
    std = np.std(image, axis=-1)
    std = std[mask]
    if len(std) == 0:
        return None
    else:
        return std.mean()

def dataset_analysis(dataset_name, threshold=8):
    rains, gts = parse_dataset(dataset_name=dataset_name)
    gt_entropies = dict()
    rain_entropies = []
    densities = []
    y_gradients = []
    gt_illus = []
    rain_illus = []
    non_monos = []
    psnrs = []
    for rain_img, gt_img in tqdm(zip(rains, gts), ncols=80):
        gt_name = gt_img.split("/")[-1]
        rain, gt = cv2.imread(rain_img).astype(np.float32), cv2.imread(gt_img).astype(np.float32)
        mask = np.abs((rain - gt)).max(axis=-1) > threshold  # obtain mask
        gt_entropies[gt_name] = gt_entropies.get(gt_name, entropy(gt))
        rain_entropies.append(entropy(rain))
        densities.append(cal_density(rain, gt))
        y_grad = cal_y_gradients(rain - gt, mask)
        if y_grad:
            y_gradients.append(y_grad)
        gt_illus.append(illuminance(gt))
        rain_illus.append(illuminance(rain))
        non_mono = non_monochromatic(rain - gt, mask)
        if non_mono:
            non_monos.append(non_mono)
        rain = torch.from_numpy(rain).permute(2, 0, 1).unsqueeze(0).float()
        gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float()
        psnrs.append(psnr(rgb_to_y(rain), rgb_to_y(gt)))
    
    gt_entropies = list(gt_entropies.values())
    

    np.savez(os.path.join("properties", "{}_prop.npz".format(dataset_name)), {
        "gt_entropies":np.array(gt_entropies),
        "rain_entropies":np.array(rain_entropies),
        "densities":np.array(densities),
        "y_gradients":np.array(y_gradients),
        "gt_illus":np.array(gt_illus),
        "rain_illus": np.array(rain_illus),
        "non_monos":np.array(non_monos),
        "psnrs": np.array(psnrs)
    })


if __name__ == "__main__":
    # dataset_analysis(dataset_name="SynRain")
    # dataset_analysis(dataset_name="GT-Rain")
    dataset_analysis(dataset_name="GTAV-balance")


