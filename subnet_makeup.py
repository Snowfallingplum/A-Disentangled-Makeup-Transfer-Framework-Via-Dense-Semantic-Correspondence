import networks
from networks import init_net
import torch
import os
import torch.nn as nn
from subnet_semantic import SFNet
import torch.nn.functional as F
class MakeupGAN(nn.Module):
    def __init__(self, opts,semantic_opts):
        super(MakeupGAN, self).__init__()
        self.opts=opts

        # parameters
        self.lr = opts.lr
        self.batch_size = opts.batch_size

        self.gpu = torch.device('cuda:{}'.format(opts.gpu)) if opts.gpu >= 0 else torch.device('cpu')
        self.input_dim = opts.input_dim
        self.output_dim = opts.output_dim
        # self.makeup_weight = opts.makeup_weight
        # self.rec_weight = opts.rec_weight
        self.CP_weight = opts.CP_weight
        self.GP_weight = opts.GP_weight
        self.cycle_weight = opts.cycle_weight
        self.adv_weight = opts.adv_weight

        # discriminators
        self.dis_non_makeup = init_net(
            networks.MultiScaleDis(opts.input_dim, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm),
            opts.gpu, init_type='normal', gain=0.02)
        self.dis_makeup = init_net(
            networks.MultiScaleDis(opts.input_dim, opts.dis_scale, norm=opts.dis_norm, sn=opts.dis_spectral_norm),
            opts.gpu, init_type='normal', gain=0.02)

        # encoders
        self.enc_c = init_net(networks.E_content(opts.input_dim), opts.gpu, init_type='normal', gain=0.02)
        self.enc_a = init_net(networks.E_attr(opts.input_dim), opts.gpu, init_type='normal', gain=0.02)

        # generator
        self.gen = init_net(networks.G(opts.input_dim, num_residule_block=opts.num_residule_block),
                            opts.gpu, init_type='normal', gain=0.02)
        # optimizers
        self.dis_non_makeup_opt = torch.optim.Adam(self.dis_non_makeup.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                                   weight_decay=0.0001)
        self.dis_makeup_opt = torch.optim.Adam(self.dis_makeup.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                               weight_decay=0.0001)
        self.enc_c_opt = torch.optim.Adam(self.enc_c.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                          weight_decay=0.0001)
        self.enc_a_opt = torch.optim.Adam(self.enc_a.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                          weight_decay=0.0001)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.5, 0.999),
                                        weight_decay=0.0001)
        self.criterionL1 = nn.L1Loss()

        # load subnet_semantic
        self.net_semantic=SFNet(semantic_opts.feature_h, semantic_opts.feature_w,
                                beta=semantic_opts.beta, kernel_sigma=semantic_opts.kernel_sigma)
        self.net_semantic.to(self.gpu)

        best_weights = torch.load(os.path.join(semantic_opts.checkpoint_dir, str(semantic_opts.epochs) + '_semantic_checkpoint.pt'))
        adap3_dict = best_weights['state_dict1']
        adap4_dict = best_weights['state_dict2']
        self.net_semantic.adap_layer_feat3.load_state_dict(adap3_dict)
        self.net_semantic.adap_layer_feat4.load_state_dict(adap4_dict)
        # The semantic subnetwork gradient is not updatable
        for param in self.net_semantic.parameters():
            param.requires_grad = False
        # Because there is a Batchnorm so net.eval
        self.net_semantic.eval()

    def forward(self):
        self.z_non_makeup_c = self.enc_c(self.non_makeup)
        self.z_non_makeup_a = self.enc_a(self.non_makeup)
        self.z_makeup_c = self.enc_c(self.makeup)
        self.z_makeup_a = self.enc_a(self.makeup)
        # sf_sub
        warp_data = self.net_semantic(self.non_makeup_norm, self.makeup_norm, train=False)
        grid_S2T1 = warp_data['grid_S2T'].permute(0, 3, 1, 2)
        grid_T2S1 = warp_data['grid_T2S'].permute(0, 3, 1, 2)
        grid_S2T1 = F.interpolate(grid_S2T1, size=(self.opts.crop_size//4,self.opts.crop_size//4),
                                  mode='bilinear', align_corners=True)
        grid_T2S1 = F.interpolate(grid_T2S1, size=(self.opts.crop_size//4,self.opts.crop_size//4),
                                  mode='bilinear', align_corners=True)
        grid_S2T2 = F.interpolate(grid_S2T1, size=(self.opts.crop_size, self.opts.crop_size),
                                  mode='bilinear', align_corners=True)
        grid_T2S2 = F.interpolate(grid_T2S1, size=(self.opts.crop_size, self.opts.crop_size),
                                  mode='bilinear', align_corners=True)
        grid_S2T1 = grid_S2T1.permute(0, 2, 3, 1)
        grid_T2S1 = grid_T2S1.permute(0, 2, 3, 1)
        grid_S2T2 = grid_S2T2.permute(0, 2, 3, 1)
        grid_T2S2 = grid_T2S2.permute(0, 2, 3, 1)
        self.makeup_semantic=F.grid_sample(self.makeup, grid_S2T2, mode='bilinear')
        self.non_makeup_semantic = F.grid_sample(self.non_makeup, grid_T2S2, mode='bilinear')

        self.z_makeup_a = F.grid_sample(self.z_makeup_a, grid_S2T1, mode='bilinear')
        self.z_non_makeup_a = F.grid_sample(self.z_non_makeup_a, grid_T2S1, mode='bilinear')


        # makeup transfer and removal
        self.z_transfer = self.gen(self.z_non_makeup_c, self.z_makeup_a)
        self.z_removal = self.gen(self.z_makeup_c, self.z_non_makeup_a)
        # cycle
        self.z_transfer_c = self.enc_c(self.z_transfer)
        self.z_transfer_a = self.enc_a(self.z_transfer)
        self.z_removal_c = self.enc_c(self.z_removal)
        self.z_removal_a = self.enc_a(self.z_removal)

        self.z_removal_a = F.grid_sample(self.z_removal_a, grid_S2T1, mode='bilinear')
        self.z_transfer_a = F.grid_sample(self.z_transfer_a, grid_T2S1, mode='bilinear')


        self.z_cycle_non_makeup = self.gen(self.z_transfer_c, self.z_removal_a)
        self.z_cycle_makeup = self.gen(self.z_removal_c, self.z_transfer_a)


    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir)
        # weight
        if train:
            self.dis_non_makeup.load_state_dict(checkpoint['dis_non_makeup'])
            self.dis_makeup.load_state_dict(checkpoint['dis_makeup'])
        self.enc_c.load_state_dict(checkpoint['enc_c'])
        self.enc_a.load_state_dict(checkpoint['enc_a'])
        self.gen.load_state_dict(checkpoint['gen'])
        # optimizer
        if train:
            self.dis_non_makeup_opt.load_state_dict(checkpoint['dis_non_makeup_opt'])
            self.dis_makeup_opt.load_state_dict(checkpoint['dis_makeup_opt'])
            self.enc_c_opt.load_state_dict(checkpoint['enc_c_opt'])
            self.enc_a_opt.load_state_dict(checkpoint['enc_a_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def save(self, filename, ep, total_it):
        state = {
            'dis_non_makeup': self.dis_non_makeup.state_dict(),
            'dis_makeup': self.dis_makeup.state_dict(),
            'enc_c': self.enc_c.state_dict(),
            'enc_a': self.enc_a.state_dict(),
            'gen': self.gen.state_dict(),
            'dis_non_makeup_opt': self.dis_non_makeup_opt.state_dict(),
            'dis_makeup_opt': self.dis_makeup_opt.state_dict(),
            'enc_c_opt': self.enc_c_opt.state_dict(),
            'enc_a_opt': self.enc_a_opt.state_dict(),
            'gen_opt': self.gen_opt.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_non_makeup = self.normalize_image(self.non_makeup).detach()
        images_makeup = self.normalize_image(self.makeup).detach()
        images_non_makeup_semantic = self.normalize_image(self.non_makeup_semantic).detach()
        images_makeup_semantic = self.normalize_image(self.makeup_semantic).detach()
        images_z_transfer = self.normalize_image(self.z_transfer).detach()
        images_z_removal = self.normalize_image(self.z_removal).detach()
        images_transfer = self.normalize_image(self.transfer).detach()
        images_removal = self.normalize_image(self.removal).detach()
        images_cycle_non_makeup = self.normalize_image(self.z_cycle_non_makeup).detach()
        images_cycle_makeup = self.normalize_image(self.z_cycle_makeup).detach()
        row1 = torch.cat(
            (images_non_makeup[0:1, ::],images_makeup_semantic[0:1, ::], images_z_transfer[0:1, ::], images_transfer[0:1, ::],
             images_cycle_non_makeup[0:1, ::]), 3)
        row2 = torch.cat(
            (images_makeup[0:1, ::],images_non_makeup_semantic[0:1, ::], images_z_removal[0:1, ::], images_removal[0:1, ::],
             images_cycle_makeup[0:1, ::]), 3)
        return torch.cat((row1, row2), 2)

    def normalize_image(self, x):
        return x[:, 0:3, :, :]

    def test_pair_forward(self,data):
        self.non_makeup = data['non_makeup'].to(self.gpu).detach()
        self.makeup = data['makeup'].to(self.gpu).detach()
        self.non_makeup_norm = data['non_makeup_norm'].to(self.gpu).detach()
        self.makeup_norm = data['makeup_norm'].to(self.gpu).detach()
        with torch.no_grad():
            self.z_non_makeup_c = self.enc_c(self.non_makeup)
            self.z_non_makeup_a = self.enc_a(self.non_makeup)
            self.z_makeup_c = self.enc_c(self.makeup)
            self.z_makeup_a = self.enc_a(self.makeup)
            # sf_sub
            warp_data = self.net_semantic(self.non_makeup_norm, self.makeup_norm, train=False)
            grid_S2T1 = warp_data['grid_S2T'].permute(0, 3, 1, 2)
            grid_T2S1 = warp_data['grid_T2S'].permute(0, 3, 1, 2)
            #print(grid_S2T1)
            grid_S2T1 = F.interpolate(grid_S2T1, size=(self.opts.crop_size // 4, self.opts.crop_size // 4),
                                      mode='bilinear', align_corners=True)
            grid_T2S1 = F.interpolate(grid_T2S1, size=(self.opts.crop_size // 4, self.opts.crop_size // 4),
                                      mode='bilinear', align_corners=True)
            grid_S2T2 = F.interpolate(grid_S2T1, size=(self.opts.crop_size, self.opts.crop_size),
                                      mode='bilinear', align_corners=True)
            grid_T2S2 = F.interpolate(grid_T2S1, size=(self.opts.crop_size, self.opts.crop_size),
                                      mode='bilinear', align_corners=True)
            grid_S2T1 = grid_S2T1.permute(0, 2, 3, 1)
            grid_T2S1 = grid_T2S1.permute(0, 2, 3, 1)
            grid_S2T2 = grid_S2T2.permute(0, 2, 3, 1)
            grid_T2S2 = grid_T2S2.permute(0, 2, 3, 1)
            self.makeup_semantic = F.grid_sample(self.makeup, grid_S2T2, mode='bilinear')
            self.non_makeup_semantic = F.grid_sample(self.non_makeup, grid_T2S2, mode='bilinear')

            self.z_makeup_a = F.grid_sample(self.z_makeup_a, grid_S2T1, mode='bilinear')
            self.z_non_makeup_a = F.grid_sample(self.z_non_makeup_a, grid_T2S1, mode='bilinear')

            # makeup transfer and removal
            self.z_transfer = self.gen(self.z_non_makeup_c, self.z_makeup_a)
            self.z_removal = self.gen(self.z_makeup_c, self.z_non_makeup_a)
            # cycle
            self.z_transfer_c = self.enc_c(self.z_transfer)
            self.z_transfer_a = self.enc_a(self.z_transfer)
            self.z_removal_c = self.enc_c(self.z_removal)
            self.z_removal_a = self.enc_a(self.z_removal)

            self.z_removal_a = F.grid_sample(self.z_removal_a, grid_S2T1, mode='bilinear')
            self.z_transfer_a = F.grid_sample(self.z_transfer_a, grid_T2S1, mode='bilinear')

            self.z_cycle_non_makeup = self.gen(self.z_transfer_c, self.z_removal_a)
            self.z_cycle_makeup = self.gen(self.z_removal_c, self.z_transfer_a)
    def test_pair_outputs(self):
        images_non_makeup = self.normalize_image(self.non_makeup).detach()
        images_makeup = self.normalize_image(self.makeup).detach()
        images_non_makeup_semantic = self.normalize_image(self.non_makeup_semantic).detach()
        images_makeup_semantic = self.normalize_image(self.makeup_semantic).detach()
        images_z_transfer = self.normalize_image(self.z_transfer).detach()
        images_z_removal = self.normalize_image(self.z_removal).detach()
        images_cycle_non_makeup = self.normalize_image(self.z_cycle_non_makeup).detach()
        images_cycle_makeup = self.normalize_image(self.z_cycle_makeup).detach()
        # row1 = torch.cat(
        #     (images_non_makeup[0:1, ::], images_makeup[0:1, ::], images_z_transfer[0:1, ::]), 3)
        row1 = torch.cat(
            (images_non_makeup[0:1, ::],images_makeup_semantic[0:1, ::], images_z_transfer[0:1, ::],  images_cycle_non_makeup[0:1, ::]), 3)
        row2 = torch.cat(
            (images_makeup[0:1, ::],images_non_makeup_semantic[0:1, ::], images_z_removal[0:1, ::],  images_cycle_makeup[0:1, ::]), 3)
        row1=torch.cat((images_non_makeup[0:1, ::],images_makeup[0:1, ::],images_z_transfer[0:1, ::]),3)
        return row1




