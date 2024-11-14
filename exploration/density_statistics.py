import cv2
import matplotlib.pyplot as plt
from parse_data import parse_dataset
import numpy as np
from tqdm import tqdm
import os


def calc_densities(dataset_name, threshold=8 / 255.0):  # rain map
    rains, gts = parse_dataset(dataset_name)
    densities = []
    for rain_img, gt_img in tqdm(zip(rains, gts), ncols=80):
        # calculate rain map
        rain, gt = cv2.imread(rain_img) / 255.0, cv2.imread(gt_img) / 255.0
        diff = np.abs(rain - gt).max(axis=-1) 
        rho = diff / (rain.max(axis=-1) + gt.max(axis=-1) + 1e-3)
        rho = rho.mean()
        densities.append(rho)
    densities = np.array(densities, dtype=np.float32)
    return densities

def calc_intensities(dataset_name, threshold=8 / 255.0):  # calculate rain intensity using linear model (Ref: weatherstream iccv2023)
    rains, gts = parse_dataset(dataset_name)
    den_intensities = np.empty((len(rains), 2), dtype=np.float32)  # densities, and intensties
    idx = 0
    for rain_img, gt_img in tqdm(zip(rains, gts), ncols=80):
        # calculate rain map
        rain, gt = cv2.imread(rain_img) / 255.0, cv2.imread(gt_img) / 255.0
        delta_I = rain - gt
        mask = (delta_I).max(axis=-1) > threshold
        rain_pixels = rain[mask].transpose((1, 0)).reshape(-1)[..., None]  # [3 * M, 1]
        gt_pixels = gt[mask].transpose((1, 0)).reshape(-1)[..., None] # [3 * M, 1]
        bias = np.ones_like(gt_pixels)
        gt_pixels = np.concatenate([gt_pixels, bias], axis=-1)  # [3, M, 2]
        gt_cor = np.matmul(gt_pixels.transpose(1, 0), gt_pixels) # [3, 2, 2]
        regression_res = np.matmul(np.linalg.inv(gt_cor), np.matmul(gt_pixels.transpose(1, 0), rain_pixels))  # linear regresssion
        intensity = regression_res[-1]
        den_intensities[idx, 0] = mask.mean()
        den_intensities[idx, 1] = intensity
        idx += 1
        if intensity < 0:
            print(rain_img)
        if intensity > 0.6:
            print(rain_img, intensity)
            print(regression_res)
    return den_intensities


def statistics(min_val, max_val, densities, num_intervals):
    intervals = np.linspace(min_val, max_val, num_intervals)
    cnts, inters = [], []
    for i in range(len(intervals) - 1):
        m = (densities >= intervals[i]) * (densities < intervals[i+1])
        cnts.append(m.astype(np.float32).sum())
        inters.append((intervals[i], intervals[i+1]))
    cnts = np.array(cnts)
    return inters, cnts


if __name__ == "__main__":
    dataset_names = ["SynRain", "GT-Rain", "GTAV-balance", "GTAV-crop", "GTAV-NightRain"]    
    fig, ax = plt.subplots(layout="constrained", figsize=(9, 3))
    width = 0.17
    colors = ["#8983BF", "#54B345", "#F27970", "cyan", "purple"]
    for multiplier, dataset_name in enumerate(dataset_names):
        if os.path.exists("{}_density.npy".format(dataset_name)):
            densities = np.load("{}_density.npy".format(dataset_name))
        else:
            densities = calc_densities(dataset_name)
            if len(densities) > 0:
                np.save("{}_density.npy".format(dataset_name), densities)
            else:
                print("No image is found ... ")
        print(densities.min(), densities.max(), densities.shape[0])
        """
        if dataset_name  == "GTAV-crop":
            densities_lowerest = densities[densities < 0.04]
            densities_lowerest = np.sort(densities_lowerest)
            densities_lowerest = densities[-15000:]
            densities = np.concatenate([densities_lowerest, densities[densities >= 0.04]])
        """
        inters, cnts = statistics(min_val=0.0, max_val=0.4, densities=densities, num_intervals=11)
        print(inters)
        print(cnts)
        x = np.arange(len(inters))
        legends = ["{:.2f}~{:.2f}".format(item[0], item[1]) for item in inters]
        rects = ax.bar(x + (multiplier-2) * width, cnts / cnts.sum(), width=width, color=colors[multiplier],
                       label=dataset_name)
    ax.set_xlabel("rain density @ illuminance interval", fontdict={"fontsize": 12})
    ax.set_ylabel("proportion", fontdict={"fontsize": 12})
    ax.legend(loc="best", ncols=3)
    ax.set_xticks(x)
    ax.set_xticklabels(legends, rotation=20)
    ax.set_ylim(bottom=0.0, top=1.0)
    plt.grid(axis="y")
    # plt.savefig("proportion_all_rela.pdf".format(dataset_name), dpi=400, pad_inches=0.1, bbox_inches="tight")
    plt.show()



