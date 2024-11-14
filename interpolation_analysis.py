from models.Anyrain_former_resize import DRSformer
from utils.parse_config import parse
import torch
import torchvision as TV
import os
import cv2

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize_img = cv2.resize(img, (128, 128))
    inp_img = torch.from_numpy(img).permute(2, 0, 1).contiguous().unsqueeze(0).float() / 255.0
    resize_img = torch.from_numpy(resize_img).permute(2, 0, 1).contiguous().unsqueeze(0).float() / 255.0
    h, w = inp_img.shape[-2:]
    h1, w1 = (h // 8) * 8, (w // 8) * 8
    inp_img = inp_img[:,:,:h1,:w1] # divided by 8
    return inp_img, resize_img

# load model
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda")
opt = parse()
model = DRSformer(opt.model).to(device)
model.load_state_dict(torch.load(opt.checkpoint))

# save_path
save_path = "interpolation/illu"
os.makedirs(save_path, exist_ok=True)

# illu
inp_img = "/home1/zhangsy/rh/data/derain/Rain800/test/rain/rain-002.png"
ref_img = "/home1/zhangsy/rh/data/derain/Rain200L/train/rain/norain-1797x2.png"

# fog
# inp_img = "/home1/zhangsy/rh/data/derain/GT-Rain/GT-RAIN_train/rain"
# ref_img = "/home1/zhangsy/rh/data/derain/GTAV-balance/train/rainy/0004_00_set1_0crop.png"


inp_img, resize_inp = read_img(inp_img)
ref_img, ref_inp = read_img(ref_img)

inp_img, resize_inp = inp_img.to(device), resize_inp.to(device)
ref_img, ref_inp = ref_img.to(device), ref_inp.to(device)

TV.utils.save_image(inp_img, os.path.join(save_path, "input.png"))
TV.utils.save_image(ref_img, os.path.join(save_path,"ref.png"))

with torch.no_grad():
    for lambda_val in [0.0]:
        out = model.interpolation(inp_img, resize_inp, ref_inp, lambda_val=lambda_val)
        out = out.clamp_(0.0, 1.0).cpu()
        TV.utils.save_image(out, os.path.join(save_path, "input{}_interpolate.png".format(lambda_val)))

        out_2 = model.interpolation(ref_img, ref_inp, resize_inp, lambda_val=lambda_val)
        out_2 = out_2.clamp_(0.0, 1.0).cpu()
        TV.utils.save_image(out_2, os.path.join(save_path, "ref{}_interpolate.png".format(lambda_val)))