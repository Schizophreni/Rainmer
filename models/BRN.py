import torch
import torch.nn as nn
import torch.nn.functional as F


class BRN(nn.Module):    
    def __init__(self, opt):
        super(BRN, self).__init__()
        self.iteration = opt.model.inter_iter
        self.use_GPU = opt.train.use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )

        self.conv0_r = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        
        self.res_conv3_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        """
        self.res_conv4_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5_r = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        """
        self.conv_i_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o_r = nn.Sequential(
            nn.Conv2d(32 + 32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_r = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )


    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        
        x = input
        r = input
        h = torch.zeros(batch_size, 32, row, col)
        c = torch.zeros(batch_size, 32, row, col)
        h_r = torch.zeros(batch_size, 32, row, col)
        c_r = torch.zeros(batch_size, 32, row, col)
        if self.use_GPU:
            h = h.cuda()
            c = c.cuda()
            h_r = h_r.cuda()
            c_r = c_r.cuda()

        # x_list = []
        # r_list = []
        for i in range(self.iteration):
            r = torch.cat((input, r), 1)
            r = self.conv0_r(r)
            r = torch.cat((r, h_r, h), 1)
            i_r = self.conv_i_r(r)
            f_r = self.conv_f_r(r)
            g_r = self.conv_g_r(r)
            o_r = self.conv_o_r(r)
            c_r = f_r * c_r + i_r * g_r
            h_r = o_r * F.tanh(c_r)
            resr = h_r
            r = F.relu(self.res_conv1_r(h_r) + resr)
            resr = r
            r = F.relu(self.res_conv2_r(r) + resr)
            resr = r
            r = F.relu(self.res_conv3_r(r) + resr)
           

            r = self.conv_r(r)
            # r_list.append(r)

            x = torch.cat((input, x, r), 1)
            x = self.conv0(x)
            x = torch.cat((x, h, h_r), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            resx = h
            x = F.relu(self.res_conv1(h) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)

            x = self.conv(x)
            # x_list.append(x)

        return x# , x_list, r, r_list
