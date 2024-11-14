"""
Train DRSformer with mixed datasets following CoIC
"""
import sys
sys.path.append("..")
from tensorboardX import SummaryWriter
import numpy as np
import os
import torch
from torch.backends import cudnn
import random
import torch.nn.functional as F
from evaluation import psnr as compare_psnr
import shutil
from torch.optim.lr_scheduler import MultiStepLR
import math
import time
from utils.parse_config import parse
import importlib
from SSIM import SSIM


def obtain_loaders(opt):
    train_opt, val_opt = opt.train, opt.val
    # train_loader
    train_loader = importlib.import_module(train_opt.dataloader.split("-")[0].strip())  # obtain module
    train_loader = getattr(train_loader, train_opt.dataloader.split("-")[-1].strip())(train_opt)
    # val_loader
    val_loader = importlib.import_module(val_opt.dataloader.split("-")[0].strip()) # obtain module
    val_loader = getattr(val_loader, val_opt.dataloader.split("-")[-1].strip())(val_opt) # instantiate validation loader 
    return train_loader, val_loader  

def loss_func(b_0, list_b, list_r, input, target):
    loss_bs = torch.stack([F.mse_loss(list_b[i], target) for i in range(opt.model.inter_iter)]).sum()
    loss_rs = torch.stack([F.mse_loss(list_r[i], input-target) for i in range(opt.model.inter_iter)]).sum()
    loss_b = F.mse_loss(list_b[-1], target)
    loss_r = F.mse_loss(list_r[-1], input-target)
    loss_b0 = F.mse_loss(b_0, target)
    return 0.1*loss_b0+0.1*loss_bs+loss_b+0.1*loss_rs+0.9*loss_r 

class Experiments:
    def __init__(self, opt):
        ## Initialize dataloader
        self.opt = opt  # obtain options from configuration file under folder configs/
        self.train_loader, self.val_loader = obtain_loaders(self.opt.datasets)
        self.epochs = (opt.train.total_iters // len(self.train_loader)) + 1
        self.batch_size = opt.datasets.train.batch_size
        if opt.train.use_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.train.gpu_id
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")
        print('# of training samples: %d \n' % int(len(self.train_loader.dataset)))
        # Build Model
        # instantiate model
        model = importlib.import_module(opt.model.model.split("-")[0].strip())  # import module
        self.model = getattr(model, opt.model.model.split("-")[-1].strip())(opt.model)  # instantiate model
        self.model.to(self.device)
        # criterion
        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt.train.optim.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=opt.train.scheduler.milestones, gamma=0.2)
        # Create log folder
        self.save_path = os.path.join(opt.log.save_path.strip(), opt.exp_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.writter = SummaryWriter(logdir=self.save_path)
        self.writter.add_text(tag="opt", text_string=str(opt))
        self.init_epoch = 1
        # save files
        shutil.copy("../"+opt.datasets.train.dataloader.split("-")[0].strip().replace(".", "/")+".py", os.path.join(self.save_path, "dataset.py"))
        shutil.copy(__file__, os.path.join(self.save_path, "train.py"))
        shutil.copy("../"+os.path.join("configs", opt.config_name), os.path.join(self.save_path, "config.yaml"))
        # Load latest checkpoint if exists
        if os.path.exists(os.path.join(self.save_path, 'latest.tar')):
            self.init_epoch = self.load_checkpoint(os.path.join(self.save_path, 'latest.tar'))
    
    def load_checkpoint(self, ckp_path):
        """
        Load checkpoint from ckp_path
        :param: obtain_epoch: obtain current_epoch in last training process if interrupted
        """
        ckp = torch.load(ckp_path)
        self.model.load_state_dict(ckp['model'])
        self.optimizer.load_state_dict(ckp['optim'])
        return int(ckp['epoch']) + 1
    
    def train(self):
        # Start training
        step = (self.init_epoch - 1)*len(self.train_loader)
        for epoch in range(self.init_epoch, self.epochs + 1):
            for param_group in self.optimizer.param_groups:
                self.writter.add_scalar(tag="lr", scalar_value=param_group["lr"], global_step=1+epoch)
            self.model.train()
            tic = time.time()
            for iter, (input_train, target_train) in enumerate(self.train_loader):
                step += 1
                if step > opt.train.total_iters:
                    epoch = self.epochs + 1
                    torch.save(self.model.state_dict(), os.path.join(self.self.save_path, 'net_latest.pth'))
                    break
                # prepare data
                input_train = input_train.to(self.device, non_blocking=True).float()   # 255 pixel value
                target_train = target_train.to(self.device, non_blocking=True).float()  # 255 pixel value
                self.model.zero_grad()
                self.optimizer.zero_grad()
                # forward
                b_0, list_b, list_r = self.model(input_train)
                pixel_loss = loss_func(b_0, list_b, list_r, input_train, target_train)
                pixel_loss.backward()
                self.optimizer.step()
                if step % self.opt.log.print_freq == 0:
                    toc = time.time()
                    out_train = torch.clamp(list_b[-1].detach().cpu(), 0.0, 255.0)
                    psnr_train = compare_psnr(out_train, target_train.cpu(), data_range=255.0)
                    self.writter.add_scalar("pixel_loss", pixel_loss.item(), step)
                    msg = "epoch {:03d}/{:03d}, [{:03d}/{:03d}] | pixel_loss: {:5f}".format(epoch, self.epochs, iter, len(self.train_loader), pixel_loss.item())
                    msg += " | psnr: {:.4f} | time: {:.3f}s".format(psnr_train, toc-tic)
                    print(msg)
                    tic = time.time()
                if step % self.opt.log.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'net_epoch_{}.pth'.format(step)))
            self.scheduler.step()
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
            }, os.path.join(self.save_path, 'latest.tar'))

if __name__ == '__main__':
    opt = parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.train.gpu_id)    
    random.seed(opt.train.seed)
    np.random.seed(opt.train.seed)
    torch.manual_seed(opt.train.seed)
    torch.cuda.manual_seed_all(opt.train.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    exp = Experiments(opt=opt)
    exp.train()
