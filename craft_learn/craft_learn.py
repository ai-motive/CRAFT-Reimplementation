import os
import sys
import json
import pprint
import time
import argparse
import shutil
import random
import general_utils as utils
import file_utils
import coordinates as coord
import train, test


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def main_generate(ini, logger=None):
    utils.folder_exists(ini['gt_path'], create_=True)

    img_fnames = sorted(utils.get_filenames(ini['img_path'], extensions=utils.IMG_EXTENSIONS))
    ann_fnames = sorted(utils.get_filenames(ini['ann_path'], extensions=utils.META_EXTENSION))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(img_fnames)))

    for idx, img_fname in enumerate(img_fnames):
        logger.info(" [GENERATE-OCR] # Processing {} ({:d}/{:d})".format(img_fname, (idx + 1), len(img_fnames)))

        _, img_core_name, img_ext = utils.split_fname(img_fname)
        img = utils.imread(img_fname, color_fmt='RGB')

        # Load json
        ann_fname = ann_fnames[idx]
        _, ann_core_name, _ = utils.split_fname(ann_fname)
        # if ann_core_name == img_core_name + img_ext: # 뒤에 .json
        if ann_core_name == img_core_name:
            with open(ann_fname) as json_file:
                json_data = json.load(json_file)
                objects = json_data['objects']
                # pprint.pprint(objects)

        bboxes = []
        for obj in objects:
            class_name = obj['classTitle']
            if class_name in ['problem_whole', 'graph_diagrams']:
                continue

            [x1, y1], [x2, y2] = obj['points']['exterior']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            rect4 = coord.convert_rect2_to_rect4([x_min, x_max, y_min, y_max])
            bboxes.append(rect4)

        file_utils.saveResult(img_file=img_core_name, img=img, boxes=bboxes, dirname=ini['gt_path'])

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_split(ini, logger=None):
    utils.folder_exists(ini['img_path'], create_=False)
    utils.folder_exists(ini['gt_path'], create_=False)
    if utils.folder_exists(ini['train_path'], create_=False):
        print(" @ Warning: train dataset path, {}, already exists".format(ini["train_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
        shutil.rmtree(ini['train_path'])
    if utils.folder_exists(ini['test_path'], create_=False):
        print(" @ Warning: test dataset path, {}, already exists".format(ini["test_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
        shutil.rmtree(ini['test_path'])
    train_img_path, test_img_path = os.path.join(ini['train_path'], 'img/'), os.path.join(ini['test_path'], 'img/')
    train_gt_path, test_gt_path = os.path.join(ini['train_path'], 'gt/'), os.path.join(ini['test_path'], 'gt/')
    shutil.copytree(ini['img_path'], train_img_path)
    shutil.copytree(ini['img_path'], test_img_path)
    shutil.copytree(ini['gt_path'], train_gt_path)
    shutil.copytree(ini['gt_path'], test_gt_path)

    train_img_fnames = sorted(utils.get_filenames(train_img_path, extensions=utils.IMG_EXTENSIONS))
    test_img_fnames = sorted(utils.get_filenames(test_img_path, extensions=utils.IMG_EXTENSIONS))
    train_gt_fnames = sorted(utils.get_filenames(train_gt_path, extensions=utils.TEXT_EXTENSIONS))
    test_gt_fnames = sorted(utils.get_filenames(test_gt_path, extensions=utils.TEXT_EXTENSIONS))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(train_img_fnames)))
    for idx in range(len(train_img_fnames)):
        len_train = round(len(train_img_fnames) * int(ini['percent_ratio'])/100)
        if idx <= len_train:
            os.remove(test_img_fnames[idx])
            os.remove(test_gt_fnames[idx])
        else:
            os.remove(train_img_fnames[idx])
            os.remove(train_gt_fnames[idx])
    train_num = len(utils.get_filenames(train_img_path, extensions=utils.IMG_EXTENSIONS))
    test_num = len(utils.get_filenames(test_img_path, extensions=utils.IMG_EXTENSIONS))
    print(" # (train, test) = ({:d}, {:d}) -> {:d} % ".
          format(train_num, test_num, int(float(train_num) / float(train_num + test_num) * 100)))
    return True

def main_train(ini, model_dir=None, logger=None):
    train_args = ['--img_path', ini['train_img_path'],
                  '--gt_path', ini['train_gt_path'],
                  '--cuda', ini['cuda'],
                  '--model_path', ini['pretrain_model_path'],
                  '--resume', ini['resume'],
                  '--batch_size', ini['batch_size'],
                  '--learning_rate', ini['learning_rate'],
                  '--momentum', ini['momentum'],
                  '--weight_decay', ini['weight_decay'],
                  '--gamma', ini['gamma'],
                  '--num_workers', ini['num_workers'],
                  '--model_root_path', ini['model_root_path']]

    train.main(train.parse_arguments(train_args), logger=logger)
    if not model_dir:
        model_dir = max([os.path.join(ini['model_root_path'],d) for d in os.listdir(ini["model_root_path"])],
                        key=os.path.getmtime)
    else:
        model_dir = os.path.join(ini["model_root_path"], model_dir)
    model_name = os.path.basename(model_dir)
    model_name_pth = os.path.join(model_dir, model_name + ".pth")
    return True, model_name

def main_test(ini, model_dir=None, logger=None):
    if not model_dir:
        model_dir = max([os.path.join(ini['model_root_path'],d) for d in os.listdir(ini["model_root_path"])],
                        key=os.path.getmtime)
    else:
        model_dir = os.path.join(ini["model_root_path"], model_dir)
    model_name = os.path.join(model_dir, os.path.basename(model_dir))

    test_args = ['--pretrain_model_path', ini['pretrain_model_path'],
                 '--test_img_path', ini['test_img_path'],
                 '--test_gt_path', ini['test_gt_path']]

    test.test(model_path=test_args.pretrain_model_path, )


def main(args):
    ini = utils.get_ini_parameters(args.ini_fname)
    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == 'GENERATE':
        main_generate(ini['GENERATE'], logger=logger)
    elif args.op_mode == 'SPLIT':
        main_split(ini['SPLIT'], logger=logger)
    elif args.op_mode == 'TRAIN':
        main_train(ini['TRAIN'], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TEST':
        main_test(ini['TEST'], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TRAIN_TEST':
        ret, model_dir = main_train(ini['TRAIN'], model_dir=args.model_dir, logger=logger)
        main_test(ini['TEST'], model_dir, logger=logger)
        print(" # Trained model directory is {}".format(model_dir))
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", required=True, choices=['GENERATE', 'SPLIT', 'TRAIN', 'TEST', 'TRAIN_TEST'], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'SPLIT' # GENERATE / SPLIT / TRAIN / TEST / TRAIN_TEST
INI_FNAME = _this_basename_ + ".ini"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            # sys.argv.extend(["--model_dir", "./pretrain/"])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))