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
from evaluation import psnr as compare_psnr
import shutil
from torch.optim.lr_scheduler import _LRScheduler
import math
import time
from utils.parse_config import parse
import importlib
from losses import pixel_fidelity
import matplotlib.pyplot as plt


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
        self.epochs = (opt.train.total_iters // len(self.train_loader)) + 1
        self.batch_size = opt.datasets.train.batch_size
        self.device = torch.device('cuda:0') if str(opt.train.gpu_id) else torch.device('cpu')
        print('# of training samples: %d \n' % int(len(self.train_loader.dataset)))
        # Build Model
        # instantiate model
        model = importlib.import_module(opt.model.model.split("-")[0].strip())  # import module
        self.model = getattr(model, opt.model.model.split("-")[-1].strip())(opt.model)  # instantiate model
        self.model.to(self.device)
        # parse knowledge atoms
        # criterion
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=opt.train.optim.lr, betas=(0.9, 0.999), 
                                                weight_decay=opt.train.optim.weight_decay)
        self.scheduler = CosineAnnealingRestartCyclicLR(self.optimizer, periods=opt.train.scheduler.periods, 
                                                        restart_weights=opt.train.scheduler.restart_weights, 
                                                        eta_mins=opt.train.scheduler.eta_mins)
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
        step = (self.init_epoch-1)*len(self.train_loader)
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
                input_train = input_train.to(self.device, non_blocking=True).float() / 255.0
                target_train = target_train.to(self.device, non_blocking=True) / 255.0
                
                self.scheduler.step(step)
                self.optimizer.zero_grad()
                outs = self.model(input_train)
                pixel_loss = pixel_fidelity(outs, target_train)
                pixel_loss_val = pixel_loss.item()
                pixel_loss.backward()

                if opt.train.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)  # clip grad following DRSformer
                self.optimizer.step()
                if step % self.opt.log.print_freq == 0:
                    toc = time.time()
                    out_train = torch.clamp(outs.detach().cpu(), 0.0, 1.0)
                    psnr_train = compare_psnr(out_train, target_train.cpu(), data_range=1.0)
                    
                    self.writter.add_scalar("pixel_loss", pixel_loss_val, step)
                    msg = "epoch {:03d}/{:03d}, [{:03d}/{:03d}] | pixel_loss: {:5f}".format(epoch, self.epochs, iter, len(self.train_loader), pixel_loss_val)
                    msg += " | psnr: {:.4f} | time: {:.3f}s".format(psnr_train, toc-tic)
                    print(msg)
                    tic = time.time()
                # save_model
                if step % self.opt.log.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'net_iter_{}.pth'.format(step)))
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
