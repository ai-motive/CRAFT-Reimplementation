import sys
import os
import torch
import torch.utils.data as data
import argparse
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
from test import test
from data_loader import ICDAR2015

###import file#######
from mseloss import Maploss
from collections import OrderedDict
from eval.script import getresult
from craft import CRAFT
from torch.autograd import Variable
import general_utils as utils


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def init_ini(ini):
    dict = {}
    dict['cuda'] = str2bool(ini['cuda'])
    dict['pretrain_model_path'] = ini['pretrain_model_path']
    dict['resume'] = ini['resume']
    dict['train_ratio'] = float(ini['train_ratio'])
    dict['batch_size'] = int(ini['batch_size'])
    dict['learning_rate'] = float(ini['learning_rate'])
    dict['momentum'] = float(ini['momentum'])
    dict['weight_decay'] = float(ini['weight_decay'])
    dict['gamma'] = float(ini['gamma'])
    dict['num_workers'] = int(ini['num_workers'])
    return dict

def str2bool(v):
    if v == 'True':
        return True
    elif v == 'False':
        return False

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def adjust_learning_rate(optimizer, gamma, step, learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = learning_rate * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args):
    ini = utils.get_ini_parameters(args.ini_fname)
    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)
    ini = init_ini(ini['TRAIN'])
    if args.op_mode == 'train':
        utils.folder_exists(args.out_path, create_=True)
        if os.path.isdir(args.in_path):
            utils.copy_folder_structure(args.in_path, args.out_path)

        # In / Out info.
        img_dir_name = 'img'
        gt_dir_name = 'gt'
        img_path = os.path.join(args.in_path, img_dir_name + '/')
        img_fnames = utils.get_filenames(img_path, extensions=utils.IMG_EXTENSIONS)
        logger.info(" [TRAIN] # Total file number to be processed: {:d}.".format(len(img_fnames)))

        # Model info.
        model_dir, model_name, model_ext = utils.split_fname(ini['pretrain_model_path'])

        net = CRAFT(pretrained=False)
        net.load_state_dict(copyStateDict(torch.load(ini['pretrain_model_path'])))
        net = net.cuda()

        net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
        cudnn.benchmark = True
        net.train()
        target_size = int(len(img_fnames) * ini['train_ratio'])
        realdata = ICDAR2015(net, IN_PATH, img_dir=img_dir_name, gt_dir=gt_dir_name, target_size=target_size)
        real_data_loader = torch.utils.data.DataLoader(
            realdata,
            batch_size=ini['batch_size'],
            shuffle=True,
            num_workers=ini['num_workers'],
            drop_last=True,
            pin_memory=True)

        optimizer = optim.Adam(net.parameters(), lr=ini['learning_rate'], weight_decay=ini['weight_decay'])
        criterion = Maploss()
        #criterion = torch.nn.MSELoss(reduce=True, size_average=True)

        step_index = 0
        loss_time = 0
        loss_value = 0
        compare_loss = 1
        for epoch in range(1000):
            train_time_st = time.time()
            loss_value = 0
            if epoch % 50 == 0 and epoch != 0:
                step_index += 1
                adjust_learning_rate(optimizer, ini['gamma'], step_index)

            st = time.time()
            for index, (real_images, real_gh_label, real_gah_label, real_mask, _) in enumerate(real_data_loader):
                images = real_images
                gh_label = real_gh_label    # gaussian heatmap
                gah_label = real_gah_label  # gaussian affinity heatmap
                mask = real_mask            # confidence mask

                images = Variable(images.type(torch.FloatTensor)).cuda()
                gh_label = gh_label.type(torch.FloatTensor)
                gah_label = gah_label.type(torch.FloatTensor)
                gh_label = Variable(gh_label).cuda()
                gah_label = Variable(gah_label).cuda()
                mask = mask.type(torch.FloatTensor)
                mask = Variable(mask).cuda()
                # affinity_mask = affinity_mask.type(torch.FloatTensor)
                # affinity_mask = Variable(affinity_mask).cuda()

                out, _ = net(images)

                optimizer.zero_grad()

                out1 = out[:, :, :, 0].cuda()
                out2 = out[:, :, :, 1].cuda()
                loss = criterion(gh_label, gah_label, out1, out2, mask)

                loss.backward()
                optimizer.step()
                loss_value += loss.item()
                if index % 2 == 0 and index > 0:
                    et = time.time()
                    print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'.format(epoch, index, len(real_data_loader), et-st, loss_value/2))
                    loss_time = 0
                    loss_value = 0
                    st = time.time()
                # if loss < compare_loss:
                #     print('save the lower loss iter, loss:',loss)
                #     compare_loss = loss
                #     torch.save(net.module.state_dict(),
                #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth')

            print('Saving state, iter:', epoch)
            rst_model_dir = os.path.join(OUT_PATH, 'model')
            rst_model_path = os.path.join(rst_model_dir, model_name + '_' + repr(epoch) + model_ext)
            utils.folder_exists(rst_model_dir, create_=True)
            test(rst_model_path)
            # test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
            getresult()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", required=True, choices=['train', ], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--in_path", required=True, type=str, help="input file")
    parser.add_argument("--out_path", default=".", help="Output folder")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'train'
INI_FNAME = _this_basename_ + ".ini"
IN_PATH = "./data/CRAFT-pytorch/Light_SSen(top)/"
OUT_PATH = "./output/CRAFT-pytorch/Light_SSen(top)/"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            sys.argv.extend(["--in_path", IN_PATH])
            sys.argv.extend(["--out_path", OUT_PATH])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))






