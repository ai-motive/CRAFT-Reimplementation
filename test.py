"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import cv2
import numpy as np
import craft_utils
import imgproc
import file_utils
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from craft import CRAFT
from collections import OrderedDict
from eval.script import eval_dataset
from python_utils.common import general as cg


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


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

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, canvas_size, mag_ratio, poly, show_time):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def main(args, logger=None):
    # load net
    net = CRAFT(pretrained=False)     # initialize

    print('Loading weights from checkpoint {}'.format(args.model_path))
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.model_path)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.model_path, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    t = time.time()

    # load data
    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.img_path)
    est_folder = os.path.join(args.rst_path, 'est')
    mask_folder = os.path.join(args.rst_path, 'mask')
    eval_folder = os.path.join(args.rst_path, 'eval')
    cg.folder_exists(est_folder, create_=True)
    cg.folder_exists(mask_folder, create_=True)
    cg.folder_exists(eval_folder, create_=True)

    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path))
        image = imgproc.loadImage(image_path)
        # image = cv2.resize(image, dsize=(768, 768), interpolation=cv2.INTER_CUBIC) ##
        bboxes, polys, score_text = test_net(net, image,
                                             text_threshold=args.text_threshold, link_threshold=args.link_threshold,
                                             low_text=args.low_text, cuda=args.cuda,
                                             canvas_size=args.canvas_size, mag_ratio=args.mag_ratio,
                                             poly=args.poly, show_time=args.show_time)
        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = mask_folder + "/res_" + filename + '_mask.jpg'
        if not(cg.file_exists(mask_file)):
            cv2.imwrite(mask_file, score_text)

        file_utils.saveResult15(image_path, bboxes, dirname=est_folder, mode='test')

    eval_dataset(est_folder=est_folder, gt_folder=args.gt_path, eval_folder=eval_folder, dataset_type=args.dataset_type)
    print("elapsed time : {}s".format(time.time() - t))

def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')

    parser.add_argument('--model_path', default='pretrain/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument("--dataset_type", required=True, choices=['mathflat', 'ic15'], help="operation mode")
    parser.add_argument("--img_path", required=True, type=str, help="Test image file path")
    parser.add_argument("--gt_path", required=True, type=str, help="Test ground truth file path")
    parser.add_argument("--rst_path", default=".", help="Result folder")
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'TEST'
MODEL_PATH = "./pretrain/craft_mlt_25k.pth"
IMG_PATH = "./data/CRAFT-pytorch/Light_SSen(top)/test/img/"
GT_PATH = "./data/CRAFT-pytorch/Light_SSen(top)/test/gt/"
# IMG_PATH = "./data/CRAFT-pytorch/test_ssen/test/img/"
# GT_PATH = "./data/CRAFT-pytorch/test_ssen/test/gt/"
RST_PATH = "./result/"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            # sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--cuda", 'False'])
            sys.argv.extend(["--model_path", MODEL_PATH])
            sys.argv.extend(["--img_path", IMG_PATH])
            sys.argv.extend(["--gt_path", GT_PATH])
            sys.argv.extend(["--rst_path", RST_PATH])
            sys.argv.extend(["--canvas_size", '980'])
            sys.argv.extend(["--mag_ratio", '1'])

            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))