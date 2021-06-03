import sys
import os
import torch
import torch.utils.data as data
import argparse
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data_loader import ICDAR2015
from mseloss import Maploss
from collections import OrderedDict
from craft import CRAFT
from torch.autograd import Variable
from datetime import datetime
from python_utils.common import general as cg
from python_utils.image import general as ig
from python_utils.json import general as jg


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def init_ini(ini):
    dict = {}
    dict['cuda'] = str2bool(ini['cuda'])
    dict['model_path'] = ini['model_path']
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
    img_fnames = cg.get_filenames(args.img_path, extensions=ig.IMG_EXTENSIONS)
    gt_fnames = cg.get_filenames(args.gt_path, extensions=cg.TEXT_EXTENSIONS)
    img_dir, _, _ = cg.split_fname(img_fnames[0])
    gt_dir, _, _ = cg.split_fname(gt_fnames[0])
    train_dir, img_dir_name, _ = cg.split_fname(img_dir)
    _, gt_dir_name, _ = cg.split_fname(gt_dir)
    logger.info(" [TRAIN] # Total file number to be processed: {:d}.".format(len(img_fnames)))

    # Load model info.
    model_dir, model_name, model_ext = cg.split_fname(args.pretrain_model_path)
    parent_model_dir = os.path.abspath(os.path.join(model_dir, os.pardir))

    model_date = datetime.today().strftime("%y%m%d")
    rst_model_dir = os.path.join(parent_model_dir, model_date)
    cg.folder_exists(rst_model_dir, create_=True)

    device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')
    net = CRAFT(pretrained=False)
    if args.pretrain_model_path:
        net.load_state_dict(copyStateDict(torch.load(args.pretrain_model_path, map_location=device)))
    if device.type == 'cuda':
        net = net.cuda()
    else:
        net = net()
    logger.info(" [TRAIN] # Pretrained model loaded from : {}".format(args.pretrain_model_path))

    cuda_ids = [int(id) for id in args.cuda_ids]
    if device.type == 'cuda':
        if len(cuda_ids) > 1:
            net = torch.nn.DataParallel(net, device_ids=cuda_ids).cuda()
        else:
            net = torch.nn.DataParallel(net, device_ids=cuda_ids)
    else:
        net = torch.nn.DataParallel(net).to(device)

    cudnn.benchmark = True
    net.train()
    real_data = ICDAR2015(net, train_dir, img_dir=img_dir_name, gt_dir=gt_dir_name, target_size=768, viz=False)
    real_data_loader = torch.utils.data.DataLoader(
        real_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)
    logger.info(" [TRAIN] # Train img & gt loaded from : {}".format(train_dir))

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Maploss()
    #criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    # loss averager
    loss_avg = Averager()

    step_index = 0
    loss_time = 0
    loss_value = 0
    compare_loss = 1
    for epoch in range(1000000):
        # Save model path
        if args.pretrain_model_path:
            rst_model_path = os.path.join(rst_model_dir, model_name + '_' + repr(epoch) + model_ext)
            rst_json_path = os.path.join(rst_model_dir, model_name + '_' + repr(epoch) + '.json')
        else:
            rst_model_path = os.path.join(rst_model_dir, model_date + '-craft_mathflat_{}_'.format(args.tgt_class.lower()) + repr(epoch) + '.pth')
            rst_json_path = os.path.join(rst_model_dir, model_name + '-craft_mathflat_{}_'.format(args.tgt_class.lower()) + repr(epoch) + '.json')

        train_time_st = time.time()
        loss_value = 0
        loss_avg.reset()
        if epoch % args.valid_epoch == 0 and epoch != 0: # default : 50
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
            loss_avg.add(loss)
            if index % 2 == 0 and index > 0:
                et = time.time()
                logger.info(" [TRAIN] # epoch {}:({}/{}) batch || train time : {:.4f} || training loss : {:.4f}".format(epoch, index*2, len(real_data_loader)*2, et-st, loss_value/2))
                loss_time = 0
                loss_value = 0
                st = time.time()

        # Init. train info.
        rst_dict = {}
        rst_dict['result_model_path'] = rst_model_path
        rst_dict['epoch'] = epoch
        rst_dict['data_size'] = len(real_data_loader)
        rst_dict['last_loss'] = float(loss)
        rst_dict['avg_loss'] = float(loss_avg.val())

        if loss_avg.val() < compare_loss:
            print('Save the lower average loss iter, loss:', loss_avg)
            compare_loss = loss_avg.val()
            torch.save(net.module.state_dict(),
                       os.path.join(rst_model_dir, 'lower_loss.pth'))
            cg.save_dict_to_json_file(rst_dict, os.path.join(rst_model_dir, 'lower_loss.json'))
            logger.info(" [TRAIN] # Saved better model to : {}".format(os.path.join(rst_model_dir, 'lower_loss.pth')))

        # Epoch이 valid_epoch 될때마다 저장 (default : 50)
        if epoch % args.valid_epoch == 0 and epoch != 0:
            logger.info(" [TRAIN] # Saving state, iter: {}".format(epoch))
            torch.save(net.module.state_dict(), rst_model_path)
            cg.save_dict_to_json_file(rst_dict, rst_json_path)
            logger.info(" [TRAIN] # Saved model to : {}".format(rst_model_path))

        logger.info(" [TRAIN] # epoch {}:({}/{}) : training average loss : {:.3f}".format(epoch, index, len(real_data_loader), float(loss_avg.val())))

    logger.info(" [TRAIN] # Total train data processed !!!")
    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--tgt_class", required=True, choices=['TEXTLINE', 'KO', 'MATH'], help="dataset type")
    parser.add_argument("--img_path", required=True, type=str, help="Train image file path")
    parser.add_argument("--gt_path", required=True, type=str, help="Train ground truth file path")

    parser.add_argument("--pretrain_model_path", required=True, help="Pretrained model path")
    parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')

    parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
    parser.add_argument('--cuda_ids', default=True, type=list, help='Allocate GPU to train model')

    parser.add_argument('--valid_epoch', default=50, type=int, help='validation epoch for model update and save')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size of training')
    parser.add_argument('--learning_rate', default=3.2768e-5, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=32, type=int, help='Number of workers used in dataloading')    

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'TRAIN'
PRETRAIN_MODEL_PATH = "./pretrain/craft_mathflat_30k_150_50.pth"
IMG_PATH = "./data/CRAFT-pytorch/Light_SSen(top)/train/img/"
GT_PATH = "./data/CRAFT-pytorch/Light_SSen(top)/train/gt/"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            # sys.argv.extend(["--cuda", 'True'])
            # sys.argv.extend(["--cuda_ids", '0'])
            sys.argv.extend(["--pretrain_model_path", PRETRAIN_MODEL_PATH])
            sys.argv.extend(["--img_path", IMG_PATH])
            sys.argv.extend(["--gt_path", GT_PATH])
            sys.argv.extend(["--resume"])
            sys.argv.extend(["--valid_epoch", '10'])
            sys.argv.extend(["--batch_size", '2'])
            sys.argv.extend(["--learning_rate", '3.2768e-5'])
            sys.argv.extend(["--momentum", '0.9'])
            sys.argv.extend(["--weight_decay", '5e-4'])
            sys.argv.extend(["--weight_decay", '5e-4'])
            sys.argv.extend(["--gamma", '0.1'])
            sys.argv.extend(["--num_workers", '0'])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))