import argparse
class SemanticOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="SFNet")
        self.parser.add_argument('--seed', type=int, default=None, help='random seed')
        self.parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size for training')
        self.parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
        self.parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.2, help='decaying factor')
        self.parser.add_argument('--decay_schedule', type=str, default='30', help='learning rate decaying schedule')
        self.parser.add_argument('--num_workers', type=int, default=8, help='number of workers for data loader')
        self.parser.add_argument('--feature_h', type=int, default=20, help='height of feature volume')
        self.parser.add_argument('--feature_w', type=int, default=20, help='width of feature volume')
        self.parser.add_argument('--train_image_path', type=str, default='../corr_datasets/images/makeup/*.*',
                                 help='directory of  images')
        self.parser.add_argument('--train_mask_path', type=str, default='../corr_datasets/segs/makeup/*.*',
                                 help='directory of  foreground masks')
        self.parser.add_argument('--beta', type=float, default=50,
                                 help='inverse temperature of softmax @ kernel soft argmax')
        self.parser.add_argument('--kernel_sigma', type=float, default=5,
                                 help='standard deviation of Gaussian kerenl @ kernel soft argmax')
        self.parser.add_argument('--lambda1', type=float, default=3, help='weight parameter of mask consistency loss')
        self.parser.add_argument('--lambda2', type=float, default=16, help='weight parameter of flow consistency loss')
        self.parser.add_argument('--lambda3', type=float, default=0.5, help='weight parameter of smoothness loss')
        self.parser.add_argument('--eval_type', type=str, default='bounding_box',
                                 choices=('bounding_box', 'image_size'),
                                 help='evaluation type for PCK threshold (bounding box | image size)')

        self.parser.add_argument('--checkpoint_dir', type=str, default='./weights',
                                 help='path for saving result images ')
        self.parser.add_argument('--log_dir', type=str, default='./logs/semantic', help='path for saving logss')
        self.parser.add_argument('--result_dir', type=str, default='./results/semantic',
                                 help='path for saving result images ')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu: e.g. 0 ,use -1 for CPU')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


class MakeupTestOptions():
    def __init__(self):
        self.parser=argparse.ArgumentParser()
        # data loader related
        self.parser.add_argument('--dataroot', type=str, default='./test/', help='path of data')
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--input_dim', type=int, default=3, help='input_dim')
        self.parser.add_argument('--output_dim', type=int, default=3, help='output_dim')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--resize_size', type=int, default=256, help='resized image size for training')
        self.parser.add_argument('--crop_size', type=int, default=256, help='cropped image size for training')
        self.parser.add_argument('--flip', type=bool, default=True, help='specified if  flipping')
        self.parser.add_argument('--nThreads', type=int, default=0, help='# of threads for data loader')

        # ouptput related
        self.parser.add_argument('--name', type=str, default='makeup', help='folder name to save outputs')
        self.parser.add_argument('--display_dir', type=str, default='./logs/makeup', help='path for saving display results')
        self.parser.add_argument('--result_dir', type=str, default='./results/makeup',
                                 help='path for saving result images and models')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./weights',
                                 help='path for saving result images ')
        self.parser.add_argument('--points_dir', type=str, default='./results/makeuppoints',
                                 help='path for saving result images ')
        self.parser.add_argument('--display_freq', type=int, default=1, help='freq (iteration) of display')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--model_save_freq', type=int, default=200, help='freq (epoch) of saving models')
        self.parser.add_argument('--display_img', type=bool, default=True, help='specified if dispaly')

        # weight
        # self.parser.add_argument('--makeup_weight', type=float, default=4, help='makeup_weight')
        # self.parser.add_argument('--rec_weight', type=float, default=0, help='rec_weight')
        self.parser.add_argument('--CP_weight', type=float, default=2, help='CP_weight')
        self.parser.add_argument('--GP_weight', type=float, default=1, help='CP_weight')
        self.parser.add_argument('--lips_weight', type=float, default=10, help='lips_weight')
        self.parser.add_argument('--eyes_weight', type=float, default=10, help='eyes_weight')
        self.parser.add_argument('--face_weight', type=float, default=0, help='face_weight')
        self.parser.add_argument('--cycle_weight', type=float, default=2, help='cycle_weight')
        self.parser.add_argument('--adv_weight', type=float, default=1, help='adv_weight')

        # training related
        self.parser.add_argument('--dis_scale', type=int, default=3, help='scale of discriminator')
        self.parser.add_argument('--dis_norm', type=str, default='None',
                                 help='normalization layer in discriminator [None, Instance]')
        self.parser.add_argument('--dis_spectral_norm', type=bool,default=True,
                                 help='use spectral normalization in discriminator')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='type of learn rate decay')
        self.parser.add_argument('--n_ep', type=int, default=600, help='number of epochs')  # 400 * d_iter
        self.parser.add_argument('--n_ep_decay', type=int, default=300,
                                 help='epoch start decay learning rate, set -1 if no decay')  # 200 * d_iter
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--num_residule_block', type=int, default=4, help='num_residule_block')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='lr')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu: e.g. 0 ,use -1 for CPU')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
