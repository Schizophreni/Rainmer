import os
import glob
from tqdm import tqdm
import cv2


datasets = {
    "SynRain": ("/data/multi-degradation/MultiRain/{}/rain", "/data/multi-degradation/MultiRain/{}/norain"),
    "GT-Rain": ("/data/GT-Rain/GT-RAIN_{}", ""),
    "GTAV-NightRain": ("/data/GTAV-NightRain", ""), 
    "GTAV-crop": ("/data/GTAV-NightRain/crops", ""),
    "GTAV-balance": ("/data/GTAV-NightRain/GTAV-balance/", "")
}

def parse_synrain(dataset_name, mode="train"):
    rain_dir, norain_dir = datasets[dataset_name]
    rain_dir, norain_dir = rain_dir.format(mode), norain_dir.format(mode)
    rains = glob.glob(os.path.join(rain_dir, "*.jpg"))
    gts = []
    for rain in rains:
        rain_name = rain.split("/")[-1]
        gts.append(os.path.join(norain_dir, rain_name))
    print("total images: ", len(rains))
    return rains, gts

def parse_gtrain(dataset_name, mode="train"):
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
        print("scene: {}, total images: {}".format(scene, len(rain_images)))
    print("total images: ", len(rains))
    return rains, gts

def parse_gtav(dataset_name, mode="train"):
    root_dir = datasets[dataset_name][0]
    all_sets = ["set1", "set2", "set3"]
    rains, gts = [], []
    for sub_set in all_sets:
        rain_images = glob.glob(os.path.join(root_dir, sub_set, mode, "rainy/*.png"))
        gt_images = []
        for rain_img in rain_images:
            rain_name = rain_img.split("/")[-1]
            gt_images.append(os.path.join(root_dir, sub_set, mode, "gt", rain_name.split("_")[0] + ".png"))
        print("subset: {}, total images: {}".format(sub_set, len(rain_images)))
        rains.extend(rain_images)
        gts.extend(gt_images)
    print("total images: ", len(rains))
    return rains, gts

def parse_gtav_crops(dataset_name, mode="train"):
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

def parse_dataset(dataset_name):
    if dataset_name == "SynRain":
        return parse_synrain(dataset_name)
    elif dataset_name == "GT-Rain":
        return parse_gtrain(dataset_name)
    elif dataset_name == "GTAV-NightRain":
        return parse_gtav(dataset_name)
    elif dataset_name in ["GTAV-crop", "GTAV-balance"]:
        return parse_gtav_crops(dataset_name)
    else:
        raise NotImplementedError

def crop_gtav(crop_size=512):
    rains, gts = parse_dataset(dataset_name="GTAV-NightRain")
    root_dir = datasets["GTAV-NightRain"][0]
    os.makedirs(os.path.join(root_dir, "crops/train/rainy"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "crops/train/gt"), exist_ok=True)
    for rain_img, gt_img in tqdm(zip(rains, gts), ncols=80):
        rain, gt = cv2.imread(rain_img), cv2.imread(gt_img)
        H, W = rain.shape[0], rain.shape[1]
        h_fold, w_fold = H // crop_size, W // crop_size
        for crop_idx in range(h_fold * w_fold):
            row_idx, col_idx = crop_idx // w_fold, crop_idx % w_fold
            crop_rain = rain[row_idx*crop_size: (row_idx+1)*crop_size, col_idx*crop_size:(col_idx+1)*crop_size, :]
            crop_gt = gt[row_idx*crop_size: (row_idx+1)*crop_size, col_idx*crop_size:(col_idx+1)*crop_size, :]
            set_idx, rain_name = rain_img.split("/")[-4], rain_img.split("/")[-1].split(".")[0]
            # print(set_idx, rain_name)
            gt_name = rain_name.split("_")[0]
            save_rain = os.path.join(root_dir, "crops/train/rainy", "{}_{}_{}crop.png".format(rain_name, set_idx, crop_idx))
            save_gt = os.path.join(root_dir, "crops/train/gt", "{}_{}_{}crop.png".format(gt_name, set_idx, crop_idx))
            cv2.imwrite(save_rain, crop_rain)
            cv2.imwrite(save_gt, crop_gt)
    print("Finished cropping ... ")
    

if __name__ == "__main__":
    # rains, gts = parse_gtav(datasets["GTAV-Nightrain"][0], mode="test")
    crop_gtav()



