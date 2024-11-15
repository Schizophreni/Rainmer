"""
Train DRSformer with mixed datasets following CoIC
"""
from tensorboardX import SummaryWriter
import argparse
import numpy as np
import os
import torch
from torch.backends import cudnn
import random
import torch.nn.functional as F
from evaluation import psnr as compare_psnr
import shutil
from torch.optim.lr_scheduler import _LRScheduler
import math
import time
from utils.parse_config import parse
import importlib
from losses import pixel_fidelity, contrastive_loss_cos, CharbonnierLoss, ssim_fidelity
import matplotlib.pyplot as plt
import torchvision as TV
import pdb
import torch._dynamo
torch._dynamo.config.suppress_errors = True


## ------------------------------------------------------------|
##   First define learning rate scheduler used in DRSformer ---|
## ----------------------------------------------------------- |

# copied from github repository: https://github.com/cschenxiang/DRSformer/tree/main/basicsr
def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.
    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i

class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    """

    def __init__(self, optimizer, periods, restart_weights=(1, ), eta_mins=(0, ), last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)), \
                'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

def obtain_loaders(opt):
    train_opt, val_opt = opt.train, opt.val
    # train_loader
    train_loader = importlib.import_module(train_opt.dataloader.split("-")[0].strip())  # obtain module
    train_loader = getattr(train_loader, train_opt.dataloader.split("-")[-1].strip())(train_opt)
    # val_loader
    val_loader = importlib.import_module(val_opt.dataloader.split("-")[0].strip()) # obtain module
    val_loader = getattr(val_loader, val_opt.dataloader.split("-")[-1].strip())(val_opt) # instantiate validation loader 
    return train_loader, val_loader   

class Experiments:
    def __init__(self, opt):
        ## Initialize dataloader
        self.opt = opt  # obtain options from configuration file under folder configs/
        self.train_loader, self.val_loader = obtain_loaders(self.opt.datasets)
        self.accumulate_grad_step = opt.train.accumulate_grad_step
        self.total_iters = self.accumulate_grad_step*opt.train.total_iters + opt.train.stage1_iters
        self.epochs = self.total_iters // len(self.train_loader) + 1
        self.batch_size = opt.datasets.train.batch_size
        if opt.train.use_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.train.gpu_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")
        print('# of training samples: %d \n' % int(len(self.train_loader.dataset)))
        # Build Model
        # instantiate model
        model = importlib.import_module(opt.model.model.split("-")[0].strip())  # import module
        self.model = getattr(model, opt.model.model.split("-")[-1].strip())(opt.model)  # instantiate model
        self.model.to(self.device)
        # self.model = torch.compile(self.model)
        # parse knowledge atoms
        self.knowledge_atoms = opt.model.knowledge_atoms
        # parse parameters
        base_params, knowledge_params = [], []
        for name, param in self.model.named_parameters():
            if "disentangle" in name:
                knowledge_params.append(param)
            else:
                base_params.append(param)
        self.base_params, self.knowledge_params = base_params, knowledge_params
        # criterion
        stage1_iter = opt.train.stage1_iters
        periods = opt.train.scheduler.periods

        periods = [item * self.accumulate_grad_step + stage1_iter for item in periods]  #  first stage base network remains untrained

        self.knowledge_optimizer = torch.optim.AdamW(params=knowledge_params, lr=opt.train.optim.lr)
        # self.knowledge_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.knowledge_optimizer,
        #                                                                       T_max=self.epochs, eta_min=1e-7)
        self.knowledge_scheduler = CosineAnnealingRestartCyclicLR(self.knowledge_optimizer, periods=periods, 
                                                        restart_weights=opt.train.scheduler.restart_weights, 
                                                        eta_mins=opt.train.scheduler.eta_mins)
        self.optimizer = torch.optim.AdamW(params=base_params, lr=opt.train.optim.lr, betas=(0.9, 0.999), 
                                            weight_decay=0)
        
        self.scheduler = CosineAnnealingRestartCyclicLR(self.optimizer, periods=periods, 
                                                        restart_weights=opt.train.scheduler.restart_weights, 
                                                        eta_mins=opt.train.scheduler.eta_mins)
        # Create log folder
        self.save_path = os.path.join(opt.log.save_path.strip(), opt.exp_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.writter = SummaryWriter(logdir=self.save_path)
        self.writter.add_text(tag="opt", text_string=str(opt))
        self.init_epoch = 1
        # save files
        shutil.copy(opt.datasets.train.dataloader.split("-")[0].strip().replace(".", "/")+".py", os.path.join(self.save_path, "dataset.py"))
        shutil.copy(opt.model.model.split("-")[0].strip().replace(".", "/")+".py", os.path.join(self.save_path, "model.py"))
        shutil.copy(__file__, os.path.join(self.save_path, "train.py"))
        shutil.copy(os.path.join("configs", opt.config_name), os.path.join(self.save_path, "config.yaml"))
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
        # ckp_dict = dict()
        # for name, param in self.model.named_parameters():
        #     if "disentangle" in name:
        #         ckp_dict[name] = ckp["model"][name]# [name.replace("_orig_mod.","")]
        #     else:
        #         ckp_dict[name] = param
        # self.model.load_state_dict(ckp_dict)
        self.optimizer.load_state_dict(ckp['optim'])
        self.knowledge_optimizer.load_state_dict(ckp['k_optim'])
        return int(ckp['epoch']) + 1

    def train(self):
        # Start training
        tot_archive = 256
        n_archives = tot_archive // self.batch_size
        charb_fidelity = CharbonnierLoss()
        if "degradation" in self.knowledge_atoms:
            rain_archives = torch.zeros(tot_archive, 3, self.opt.datasets.train.crop_size, self.opt.datasets.train.crop_size, dtype=torch.float32).cuda()  # store recent rain for models with small batch size
            rain_svds = torch.zeros(tot_archive, 3, 32).cuda()
        if "chromatic" in self.knowledge_atoms:
            rain_chromatics = torch.zeros(tot_archive, 3).cuda()
        n_recored = 0
        n_neg = self.opt.contrastive.n_neg  # number of negatives
        step = (self.init_epoch-1)*len(self.train_loader)
        # preload archives
        for (_, _, indicators, resize_inp, resize_tar, _) in self.train_loader:
            input_train = resize_inp.float() / 255.0
            target_train = resize_tar.float() / 255.0
            if "degradation" in self.knowledge_atoms:
                rain_archives[n_recored*self.batch_size:(n_recored+1)*self.batch_size] = (input_train - target_train).clamp_(0.0, 1.0)
                rain_svds[n_recored*self.batch_size:(n_recored+1)*self.batch_size] = torch.linalg.svdvals(F.interpolate((input_train-target_train).clamp_(0.0, 1.0), (32, 32)))
            if "chromatic" in self.knowledge_atoms:
                rain_chromatics[n_recored*self.batch_size:(n_recored+1)*self.batch_size] = input_train.mean(dim=[-2, -1])
            n_recored += 1
            if n_recored == n_archives:
                break
        for epoch in range(self.init_epoch, self.epochs + 1):
            for param_group in self.optimizer.param_groups:
                self.writter.add_scalar(tag="lr", scalar_value=param_group["lr"], global_step=1+epoch)
            self.model.train()
            tic = time.time()
            for iter, (input_train, target_train, indicators, resize_inp, resize_tar, blur_tar) in enumerate(self.train_loader):
                step += 1
                if step > self.total_iters:
                    epoch = self.epochs + 1
                    torch.save(self.model.state_dict(), os.path.join(self.self.save_path, 'net_latest.pth'))
                    break
                # prepare data
                input_train = input_train.to(self.device, non_blocking=True).float() / 255.0
                target_train = target_train.to(self.device, non_blocking=True) / 255.0
                resize_inp = resize_inp.to(self.device, non_blocking=True) / 255.0
                resize_tar = resize_tar.to(self.device, non_blocking=True) / 255.0
                blur_tar = blur_tar.to(self.device, non_blocking=True) / 255.0
                im_k_dict, im_negs_dict = dict(), dict()
                # postive and negative for illumination, detail, and degradation
                # detail negatives: blur images
                if "detail" in self.knowledge_atoms:
                    # retrieve clean archive to find random sample
                    im_k_dict["detail"] = resize_inp
                    im_negs_dict["detail"] = blur_tar
                
                # illumination negatives: max dissimilar mean pixel value
                if "chromatic" in self.knowledge_atoms:
                    current_chromatic = resize_inp.mean(dim=[-2, -1])  # [b, 3]
                    diff_std = (current_chromatic.unsqueeze(1) - rain_chromatics.unsqueeze(0)).abs().sum(dim=-1)  # [b, 1, 3] - [1, len(archive), 3] ==> [b, len]
                    _, top_indices = torch.topk(diff_std, k=n_neg)
                    im_k_dict["chromatic"] = self.train_loader.dataset.random_flip(resize_tar)
                    select_chromatic = rain_chromatics[top_indices]  # [b, n_neg, 3]
                    illu_adjust_coeff = (1e-5+select_chromatic) / (1e-5+current_chromatic.unsqueeze(1))
                    im_negs_dict["chromatic"] = self.train_loader.dataset.random_flip(
                        illu_adjust_coeff.unsqueeze(-1).unsqueeze(-1)*resize_inp.unsqueeze(1)
                     ).clamp_(0.0, 1.0) # [b, n, 3, 1, 1] * [b, 1, 3, h, w]
                # for idx in range(n_neg):
                #     TV.utils.save_image(im_negs_dict["chromatic"][0, idx], "debugs/chr_neg_{}.png".format(idx))
                # degradation negatives: other types
                if "degradation" in self.knowledge_atoms:
                    current_svd = torch.linalg.svdvals(F.interpolate((resize_inp - resize_tar).clamp_(0.0, 1.0), size=(32, 32)))  # [b, 3, d]
                    diff_rain = (current_svd.unsqueeze(1) - rain_svds.unsqueeze(0)).abs().sum(dim=[2, 3])  # [b, len_archive]
                    # print(diff_rain)
                    _, topk_index_rain =  torch.topk(diff_rain, k=n_neg, dim=-1)
                    max_diff_rain = rain_archives[topk_index_rain] # [B, n_neg, 3, H, W]
                    im_cross_rains = (max_diff_rain.to(self.device) + resize_tar.unsqueeze(1)).clamp_(0.0, 1.0)  # rain negatives
                    im_k_dict["degradation"] = self.train_loader.dataset.random_flip(resize_inp)
                    im_negs_dict["degradation"] = self.train_loader.dataset.random_flip(im_cross_rains)
                    # for idx in range(n_neg):
                    #    TV.utils.save_image(im_negs_dict["degradation"][0, idx], "debugs/deg_neg_{}.png".format(idx))
                    # #    TV.utils.save_image(clean_archives[topk_index_rain[0, idx]], "debugs/deg_clean_{}.png".format(idx))
                    # #    TV.utils.save_image(clean_archives[topk_index_rain[0, idx]] + rain_archives[topk_index_rain[0, idx]], "debugs/deg_rain_{}.png".format(idx))
                # pdb.set_trace()
                # update archives
                n_recored = n_recored % n_archives
                if "degradation" in self.knowledge_atoms:
                    rain_archives[n_recored*self.batch_size:(n_recored+1)*self.batch_size] = (resize_inp - resize_tar).clamp_(0.0, 1.0)
                    rain_svds[n_recored*self.batch_size:(n_recored+1)*self.batch_size] = current_svd
                if "chromatic" in self.knowledge_atoms:
                    rain_chromatics[n_recored*self.batch_size:(n_recored+1)*self.batch_size] = current_chromatic
                n_recored = (n_recored + 1) % n_archives
                # pdb.set_trace()
                self.scheduler.step(step)
                self.knowledge_scheduler.step(step)
                # do forward
                outs, logits_dict = self.model(input_train, resize_inp, im_k_dict, im_negs_dict, adapt=True)
                # compute loss
                atom_loss = dict()
                total_loss = 0.0
                # parse contra loss
                if "chromatic" in self.knowledge_atoms:
                    atom_loss["chr"] = contrastive_loss_cos(logits=logits_dict["chromatic"], weight=None)
                    total_loss += atom_loss["chr"] * self.opt.train.stage2_lambda
                if "degradation" in self.knowledge_atoms:
                    logits_deg = logits_dict["degradation"]
                    logits_det = logits_dict["detail"][:, 1:]
                    logits_deg[:, 0] += logits_dict["detail"][:, 0] 
                    logits_deg = torch.cat([logits_deg, logits_det], dim=1)  # [deg and detail share same contrastive]
                    atom_loss["deg"] = contrastive_loss_cos(logits_deg) 
                    total_loss += atom_loss["deg"] * self.opt.train.stage2_lambda
                # add pixel loss
                pixel_loss = 0.1 * ssim_fidelity(outs, target_train) + charb_fidelity(outs, target_train)
                total_loss += pixel_loss
                pixel_loss_val = pixel_loss.item()
                (total_loss / self.accumulate_grad_step).backward()
                
                # do accumulate
                if (iter + 1) % self.accumulate_grad_step == 0:
                    if opt.train.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.base_params, 0.01)  # clip grad following DRSformer
                    self.optimizer.step()
                    self.knowledge_optimizer.step()
                    # reset gradient
                    self.knowledge_optimizer.zero_grad()
                    self.optimizer.zero_grad()

                if step % self.opt.log.print_freq == 0:
                    toc = time.time()
                    out_train = torch.clamp(outs.detach().cpu(), 0.0, 1.0)
                    psnr_train = compare_psnr(out_train, target_train.cpu(), data_range=1.0)
                    self.writter.add_scalar("pixel_loss", pixel_loss_val, step)
                    msg = "epoch {:03d}/{:03d}, [{:03d}/{:03d}] | l_pix: {:5f}".format(epoch, self.epochs, iter, len(self.train_loader), pixel_loss_val)
                    for atom in atom_loss:
                        self.writter.add_scalar(f"{atom}_loss", atom_loss[atom].item(), step)
                        msg += " | l_{}: {:.5f}".format(atom[:3], atom_loss[atom].item())
                    msg += "| l_tot: {:.5f} | psnr: {:.4f} | time: {:.3f}s".format(total_loss.item(), psnr_train, toc-tic)
                    print(msg)
                    tic = time.time()
                    TV.utils.save_image(out_train, "debugs/derain_{}.png".format(self.opt.train.stage2_lambda))
                    TV.utils.save_image(input_train, "debugs/rain_{}.png".format(self.opt.train.stage2_lambda))
                    TV.utils.save_image(target_train, "debugs/gt_{}.png".format(self.opt.train.stage2_lambda))
                # save_model
                if step % self.opt.log.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'net_epoch_{}.pth'.format((step-opt.train.stage1_iters)//2)))
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optim': self.optimizer.state_dict(),
                'k_optim': self.knowledge_optimizer.state_dict(),
            }, os.path.join(self.save_path, 'latest.tar'))

if __name__ == '__main__':
    from occ_gpu import occumpy_mem
    opt = parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.train.gpu_id)    
    random.seed(opt.train.seed)
    np.random.seed(opt.train.seed)
    torch.manual_seed(opt.train.seed)
    torch.cuda.manual_seed_all(opt.train.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # occumpy_mem(str(opt.train.gpu_id), 1024*25)
    exp = Experiments(opt=opt)
    exp.train()