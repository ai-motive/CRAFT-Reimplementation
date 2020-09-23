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


def main(args, logger=None):

    # Load image & gt files
    img_fnames = utils.get_filenames(args.train_img_path, extensions=utils.IMG_EXTENSIONS)
    gt_fnames = utils.get_filenames(args.train_gt_path, extensions=utils.TEXT_EXTENSIONS)
    img_dir, _, _ = utils.split_fname(img_fnames[0])
    gt_dir, _, _ = utils.split_fname(gt_fnames[0])
    train_dir, img_dir_name, _ = utils.split_fname(img_dir)
    _, gt_dir_name, _ = utils.split_fname(gt_dir)
    logger.info(" [TRAIN] # Total file number to be processed: {:d}.".format(len(img_fnames)))

    # Load model info.
    model_dir, model_name, model_ext = utils.split_fname(args.pretrain_model_path)

    net = CRAFT(pretrained=False)
    net.load_state_dict(copyStateDict(torch.load(args.pretrain_model_path)))
    net = net.cuda()

    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    cudnn.benchmark = True
    net.train()
    real_data = ICDAR2015(net, train_dir, img_dir=img_dir_name, gt_dir=gt_dir_name, target_size=768)
    real_data_loader = torch.utils.data.DataLoader(
        real_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
            adjust_learning_rate(optimizer, args.gamma, step_index, args.learning_rate)

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
                logger.info(" [TRAIN] # epoch {}:({}/{}) batch || train time for 2 batch : {} || training loss : {}".format(epoch, index, len(real_data_loader), et-st, loss_value/2))
                loss_time = 0
                loss_value = 0
                st = time.time()
            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.module.state_dict(),
            #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth')

            # Epoch이 +50마다 저장
            if epoch % 50 == 0 and epoch != 0:
                logger.info(" [TRAIN] # Saving state, iter: {}".format(epoch))
                rst_model_dir = os.path.join(args.model_root_path, 'model')
                rst_model_path = os.path.join(rst_model_dir, model_name + '_' + repr(epoch) + model_ext)
                rst_json_path = os.path.join(rst_model_dir, model_name + '_' + repr(epoch) + '.json')
                utils.folder_exists(rst_model_dir, create_=True)

                rst_dict = {}
                rst_dict['result_model_path'] = rst_model_path
                rst_dict['epoch'] = epoch
                rst_dict['data_size'] = len(real_data_loader)
                rst_dict['last_loss'] = float(loss)

                torch.save(net.module.state_dict(), rst_model_path)
                utils.save_dict_to_json_file(rst_dict, rst_json_path)

    logger.info(" [TRAIN] # Total train data processed !!!")
    return True


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
    parser.add_argument("--pretrain_model_path", required=True, type=str, help="pretrain model path")
    parser.add_argument("--train_img_path", required=True, type=str, help="Train image file path")
    parser.add_argument("--train_gt_path", required=True, type=str, help="Train ground truth file path")
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size of training')
    parser.add_argument('--learning_rate', default=3.2768e-5, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=32, type=int, help='Number of workers used in dataloading')
    parser.add_argument("--model_root_path", default=".", help="Saved model path")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'TRAIN'
PRETRAIN_MODEL_PATH = "./pretrain/craft_mlt_25k.pth"
TRAIN_IMG_PATH = "./data/CRAFT-pytorch/Light_SSen(top)/train/img/"
TRAIN_GT_PATH = "./data/CRAFT-pytorch/Light_SSen(top)/train/gt/"
MODEL_ROOT_PATH = "./pretrain/"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--cuda", 'True'])
            sys.argv.extend(["--pretrain_model_path", PRETRAIN_MODEL_PATH])
            sys.argv.extend(["--train_img_path", TRAIN_IMG_PATH])
            sys.argv.extend(["--train_gt_path", TRAIN_GT_PATH])
            sys.argv.extend(["--resume"])
            sys.argv.extend(["--batch_size", '2'])
            sys.argv.extend(["--learning_rate", '3.2768e-5'])
            sys.argv.extend(["--momentum", '0.9'])
            sys.argv.extend(["--weight_decay", '5e-4'])
            sys.argv.extend(["--weight_decay", '5e-4'])
            sys.argv.extend(["--gamma", '0.1'])
            sys.argv.extend(["--num_workers", '0'])
            sys.argv.extend(["--model_root_path", MODEL_ROOT_PATH])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))








