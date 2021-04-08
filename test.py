import os
import torch
from dataset_makeup import MakeupDataset
from options import SemanticOptions,MakeupTestOptions
from subnet_makeup import MakeupGAN
from saver import Saver
import warnings
warnings.filterwarnings("ignore")

def pair_test():
    # parse options
    makeup_parser = MakeupTestOptions()
    makeup_opts = makeup_parser.parse()
    semantic_parser = SemanticOptions()
    semantic_opts = semantic_parser.parse()

    # data loader
    print('\n--- load dataset ---')
    dataset = MakeupDataset(makeup_opts)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=makeup_opts.batch_size, shuffle=False,
                                               num_workers=makeup_opts.nThreads)
    # model
    print('\n--- load model ---')
    model = MakeupGAN(makeup_opts, semantic_opts)

    ep0, total_it = model.resume(os.path.join(makeup_opts.checkpoint_dir,'00599.pth'),makeup_opts.phase)
    model.eval()
    print('start pair test')
    # saver for display and output
    saver = Saver(makeup_opts)
    for iter,data in enumerate(train_loader):
        with torch.no_grad():
            model.test_pair_forward(data)
            saver.write_test_pair_img(iter, model)



if __name__ == '__main__':
    pair_test()
    print('The test is complete')