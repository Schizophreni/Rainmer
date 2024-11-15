from sklearn.manifold import TSNE
import torch
from models.Rainmer_SA_chralpha import DRSformer
from tqdm import tqdm
import glob
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import umap
import numba
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datasets.dataset import prepare_gtrain, prepare_gtav_balance
# matplotlib inline

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def parse_data(dataset_name, max_num=200, sub_idx=1):  # sub_idx: sub index for Rain1200 and Rain1400
    if dataset_name == "Rain200H":
        data_dir = "/home1/zhangsy/rh/data/derain/Rain200H/train/rain/X2".format(dataset_name)
        imgs = glob.glob(os.path.join(data_dir, "*.png"))
    elif dataset_name == "Rain200L":
        data_dir = "/home1/zhangsy/rh/data/derain/Rain200L/train/rain/".format(dataset_name)
        imgs = glob.glob(os.path.join(data_dir, "*.png"))
    elif dataset_name == "Rain800":
        data_dir = "/home1/zhangsy/rh/data/derain/Rain800/train/rain/"
        imgs = glob.glob(os.path.join(data_dir, "*.png"))
    elif dataset_name == "GT-Rain":
        imgs, _ = prepare_gtrain("gt-rain", "train")
    elif dataset_name == "GTAV-balance":
        imgs, _ = prepare_gtav_balance("gtav-balance", "train")
    elif dataset_name == "DID":
        data_dir = os.path.join('/home1/zhangsy/rh/data/derain/Rain1200_new/train')
        imgs = glob.glob(os.path.join(data_dir, "*.jpg"))
        imgs = sorted(imgs, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        imgs = imgs[4000*(sub_idx-1):4000*sub_idx]
    elif dataset_name == "DDN":
        data_dir = os.path.join("/home1/zhangsy/rh/data/derain/Rain14000/train/rain")
        imgs = glob.glob(os.path.join(data_dir, "*.jpg"))
        imgs = [item for item in imgs if int(item.split("_")[-1].split(".")[0]) == sub_idx]
    elif dataset_name == "Clean":
        data_dir = "/home1/zhangsy/rh/data/derain/Rain200L/train/norain".format(dataset_name)
        imgs = glob.glob(os.path.join(data_dir, "*.png"))
    elif dataset_name == "RealInt":
        data_dir = "/home1/zhangsy/rh/data/derain/Real_Internet"
        imgs = glob.glob(os.path.join(data_dir, "*.png"))
    elif dataset_name == "Snow":
        data_dir = "/home1/zhangsy/rh/data/derain/AllinOne/snow/train/input"
        imgs = glob.glob(os.path.join(data_dir, "*.jpg"))
    elif dataset_name == "Outdoor-Rain":
        data_dir = "/home1/zhangsy/rh/data/derain/AllinOne/rain/train/input"
        imgs = glob.glob(os.path.join(data_dir, "*.png"))
    elif dataset_name == "Raindrop":
        data_dir = "/home1/zhangsy/rh/data/derain/AllinOne/raindrop/train/input"
        imgs = glob.glob(os.path.join(data_dir, "*.png"))
    imgs = np.random.choice(imgs, size=max_num)
    # imgs = imgs[:max_num]
    return imgs

def encode(model, imgs, z_type="chr"):
    transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),  # 96 for DRSformer, other: 128
    ])
    totensor = transforms.Compose([
        transforms.ToTensor()
    ])

    embed = np.empty((len(imgs), 128), dtype=np.float32)
    probs = np.empty((len(imgs), 3), dtype=np.float32)
    with torch.no_grad():
        for idx, img in tqdm(enumerate(imgs)):
            img_name = img
            img = Image.open(img).convert("RGB")
            img = totensor(img)
            if "Rain1200" in img_name:
                img, _ = torch.chunk(img, chunks=2, dim=-1)
            img = transform(img).unsqueeze(0).cuda()
            out_dict = model.disentangle.encoder_q.forward(img)
            if z_type == "chr":
                mid = out_dict["chromatic"][0].squeeze(-1).squeeze(-1)
            else:
                mid = out_dict["degradation"][0].squeeze(-1).squeeze(-1)
            # mid = z
            embed[idx] = mid.cpu().numpy()
    return embed, probs

@numba.njit()
def dist(a, b):
    s = (a*b).sum()
    s = min(max(s, -1.0), 1.0)
    return np.arccos(s)

def visualize(embeds:list, dataset_names: list):
    sns.set(style="white", rc={'figure.figsize': (8, 6)})
    classes = []
    labels = dataset_names
    embeds_len = [0]
    print(labels)
    for idx, embed in enumerate(embeds):
        classes.extend([idx for _ in range(embed.shape[0])])
        embeds_len.append(embeds_len[-1] + len(embed))
    print("embeds length: ", embeds_len)
    embeds = np.concatenate(embeds, axis=0)
    
    sphere_mapper = umap.UMAP(n_neighbors=50, metric=dist, output_metric="haversine", 
                              min_dist=0.0, random_state=42).fit(embeds)
    x = np.sin(sphere_mapper.embedding_[:, 0]) * np.cos(sphere_mapper.embedding_[:, 1])
    y = np.sin(sphere_mapper.embedding_[:, 0]) * np.sin(sphere_mapper.embedding_[:, 1])
    z = np.cos(sphere_mapper.embedding_[:, 0])
    """
    dots = np.concatenate([x[None, :], y[None, :], z[None, :]], axis=0)
    for idx in range(len(embeds_len) - 2):
        for j in range(idx, len(embeds_len)-1):
            dots_x = dots[:, embeds_len[idx]:embeds_len[idx+1]]
            dots_y = dots[:, embeds_len[j]:embeds_len[j+1]]
            sim = np.dot(dots_x.T, dots_y)
            print("{} < - > {}, sim: mean: {}, max: {}".format(idx, j, sim.mean(), sim.max()))
    """
    x = np.arctan2(x, y)
    y = np.arccos(z)

    # fig = plt.figure()
    fig, ax = plt.subplots(constrained_layout=True)
    # ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(x, y, marker=".", c=classes, cmap=plt.cm.get_cmap("jet", len(embeds)), label=labels, alpha=1.0)
    # scatter = ax.scatter(x, y, z, marker=".", c=classes, cmap="Spectral", label=labels)

    handles, _ = scatter.legend_elements(alpha=0.7)
    ax.legend(handles, labels, fontsize=10, bbox_to_anchor=(1.35, 0.0), loc="lower right", borderaxespad=0)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # ax.set_xticks(np.linspace(-1, 1, 5))
    # ax.set_yticks(np.linspace(-1, 1, 5))
    # ax.set_zticks(np.linspace(-1, 1, 5))
    plt.savefig("visualizations/umap_vis.pdf", pad_inches=0.1, bbox_inches="tight", dpi=400)

if __name__ == "__main__":
    # obtain opt and model
    from utils.parse_config import parse
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda")
    opt = parse()
    model = DRSformer(opt.model).to(device)
    # load ckp
    print("Load checkpoint ... ")
    ckp = torch.load(opt.checkpoint)
    model.load_state_dict(ckp)
    model.eval()
    dataset_names = ["Snow", "Outdoor-Rain", "Raindrop"]
    # ["Rain200H", "Rain800","DDN_1", "DDN_5", "GT-Rain", "GTAV-balance"] #  "Rain1400_3", "Rain1400_4", "Rain1400_5", "Rain1400_6", "Rain1400_7",
                    #  "Rain1400_8", "Rain1400_9", "Rain1400_10", "Rain1400_11", "Rain1400_12", "Rain1400_13", 
                     # "Rain1400_14"]
    embeds = []
    probs = []
    similarities = np.zeros((len(dataset_names), len(dataset_names)))
    for name in dataset_names:
        if "DID" in name or "DDN" in name:
            sub_idx = int(name.split("_")[-1].split(" ")[0])
            name = name.split("_")[0]
        else:
            name = name.split()[0]
            sub_idx = -1
        if not "RealInt" in name:
            imgs = parse_data(name, max_num=500, sub_idx=sub_idx)
        else:
            imgs = parse_data(name, max_num=146, sub_idx=sub_idx)
        zs, ps = encode(model, imgs, z_type="chr")
        embeds.append(zs)
        probs.append(ps)
    for i in range(len(dataset_names)):
        for j in range(i, len(dataset_names)):
            sim = np.matmul(embeds[i], embeds[j].T)
            similarities[i, j] = sim.mean()
            similarities[j, i] = sim.mean()
            print("{} <-> {}, sim: {} {} {}".format(dataset_names[i], dataset_names[j], sim.mean(), sim.min(), sim.max()))
    visualize(embeds=embeds, dataset_names=dataset_names)

    plt.clf()
    plt.subplots(figsize=(6, 6), facecolor="w")
    fig = sns.heatmap(similarities, annot=True, vmax=similarities.max(), vmin=similarities.min(), square=True, fmt=".1g", yticklabels=dataset_names, 
                      cmap="coolwarm", xticklabels=dataset_names)
    fig.get_figure().savefig('visualizations/df_corr.pdf',bbox_inches='tight',transparent=True)

    # np.save("emb.npy", np.concatenate(embeds, axis=0))
        
    

