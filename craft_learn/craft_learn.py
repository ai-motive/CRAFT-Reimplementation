import os
import sys
import json
import pprint
import time
import argparse
import shutil
import random
import torch
import general_utils as utils
import file_utils
import coordinates as coord
import train, test
import subprocess
from sklearn.model_selection import train_test_split
from easyocr.detection import get_detector, get_textbox


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def load_craft_parameters(ini):
    params = {}
    params['min_size'] = int(ini['min_size'])
    params['text_threshold'] = float(ini['text_threshold'])
    params['low_text'] = float(ini['low_text'])
    params['link_threshold'] = float(ini['link_threshold'])
    params['canvas_size'] = int(ini['canvas_size'])
    params['mag_ratio'] = float(ini['mag_ratio'])
    params['slope_ths'] = float(ini['slope_ths'])
    params['ycenter_ths'] = float(ini['ycenter_ths'])
    params['height_ths'] = float(ini['height_ths'])
    params['width_ths'] = float(ini['width_ths'])
    params['add_margin'] = float(ini['add_margin'])
    return params

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
        ann_core_name = ann_core_name.replace('.jpg', '')
        if ann_core_name == img_core_name:
            with open(ann_fname) as json_file:
                json_data = json.load(json_file)
                objects = json_data['objects']
                # pprint.pprint(objects)

        bboxes = []
        texts = []
        for obj in objects:
            class_name = obj['classTitle']
            if class_name != DATASET_TYPE.lower():
                continue

            [x1, y1], [x2, y2] = obj['points']['exterior']
            text = obj['description']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            rect4 = coord.convert_rect2_to_rect4([x_min, x_max, y_min, y_max])
            bboxes.append(rect4)
            texts.append(text)

        file_utils.saveResult(img_file=img_core_name, img=img, boxes=bboxes, texts=texts, dirname=ini['gt_path'])

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
        # shutil.rmtree(ini['train_path'])

    if utils.folder_exists(ini['test_path'], create_=False):
        print(" @ Warning: test dataset path, {}, already exists".format(ini["test_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
        # shutil.rmtree(ini['test_path'])

    train_ratio = float(ini['train_ratio'])
    test_ratio = round(1.0-train_ratio, 2)

    lower_dataset_type = DATASET_TYPE.lower() if DATASET_TYPE != 'TEXTLINE' else ''
    tgt_dir = 'craft_{}_gt'.format(lower_dataset_type)
    gt_list = sorted(utils.get_filenames(ini['gt_path'], extensions=utils.TEXT_EXTENSIONS))
    train_gt_list, test_gt_list = train_test_split(gt_list, train_size=train_ratio, random_state=2000)
    train_img_list, test_img_list = [gt_path.replace(tgt_dir, 'img').replace('.txt', '.jpg').replace('gt_', '') for gt_path in train_gt_list], \
                                    [gt_path.replace(tgt_dir, 'img').replace('.txt', '.jpg').replace('gt_', '') for gt_path in test_gt_list]

    train_img_path, test_img_path = os.path.join(ini['train_path'], 'img/'), os.path.join(ini['test_path'], 'img/')
    train_gt_path, test_gt_path = os.path.join(ini['train_path'], tgt_dir+'/'), os.path.join(ini['test_path'], tgt_dir+'/')
    utils.folder_exists(train_img_path, create_=True), utils.folder_exists(test_img_path, create_=True)
    utils.folder_exists(train_gt_path, create_=True), utils.folder_exists(test_gt_path, create_=True)

    # Apply symbolic link for gt & img path
    if len(gt_list) != 0:
        for op_mode in ['train', 'test']:
            if op_mode == 'train':
                gt_list = train_gt_list
                img_list = train_img_list

                gt_link_path = train_gt_path
                img_link_path = train_img_path
            elif op_mode == 'test':
                gt_list = test_gt_list
                img_list = test_img_list

                gt_link_path = test_gt_path
                img_link_path = test_img_path

            # link gt_path
            for gt_path in gt_list:
                gt_sym_cmd = 'ln "{}" "{}"'.format(gt_path, gt_link_path)  # to all files
                subprocess.call(gt_sym_cmd, shell=True)
            logger.info(" # Link gt files {} -> {}.".format(gt_list[0], gt_link_path))

            # link img_path
            for img_path in img_list:
                img_sym_cmd = 'ln "{}" "{}"'.format(img_path, img_link_path)  # to all files
                subprocess.call(img_sym_cmd, shell=True)
            logger.info(" # Link img files {} -> {}.".format(img_list[0], img_link_path))

    print(" # (train, test) = ({:d}, {:d}) -> {:d} % ".
          format(len(train_gt_list), len(test_gt_list), int(float(len(train_gt_list))/float(len(train_gt_list)+len(test_gt_list))*100)))
    return True


def main_merge(ini, logger=None):
    utils.folder_exists(ini['total_dataset_path'], create_=True)

    datasets = [dataset for dataset in os.listdir(ini['dataset_path']) if dataset != 'total']
    sort_datasets = sorted(datasets, key=lambda x: (int(x.split('_')[0])))

    lower_dataset_type = DATASET_TYPE.lower() if DATASET_TYPE != 'TEXTLINE' else ''
    tgt_dir = 'craft_{}_gt'.format(lower_dataset_type)
    if len(sort_datasets) != 0:
        for dir_name in sort_datasets:
            src_train_path, src_test_path = os.path.join(ini['dataset_path'], dir_name, 'train'), os.path.join(ini['dataset_path'], dir_name, 'test')
            src_train_img_path, src_train_gt_path = os.path.join(src_train_path, 'img/'), os.path.join(src_train_path, tgt_dir+'/')
            src_test_img_path, src_test_gt_path = os.path.join(src_test_path, 'img/'), os.path.join(src_test_path, tgt_dir+'/')

            dst_train_path, dst_test_path = os.path.join(ini['total_dataset_path'], 'train'), os.path.join(ini['total_dataset_path'], 'test')
            dst_train_img_path, dst_train_gt_path = os.path.join(dst_train_path, 'img/'), os.path.join(dst_train_path, tgt_dir+'/')
            dst_test_img_path, dst_test_gt_path = os.path.join(dst_test_path, 'img/'), os.path.join(dst_test_path, tgt_dir+'/')

            if utils.folder_exists(dst_train_img_path) and utils.folder_exists(dst_train_gt_path) and \
                    utils.folder_exists(dst_test_img_path) and utils.folder_exists(dst_test_gt_path):
                logger.info(" # Already {} is exist".format(ini['total_dataset_path']))
            else:
                utils.folder_exists(dst_train_img_path, create_=True), utils.folder_exists(dst_train_gt_path, create_=True)
                utils.folder_exists(dst_test_img_path, create_=True), utils.folder_exists(dst_test_gt_path, create_=True)

            # Apply symbolic link for gt & img path
            for op_mode in ['train', 'test']:
                if op_mode == 'train':
                    src_img_path, src_gt_path = src_train_img_path, src_train_gt_path
                    dst_img_path, dst_gt_path = dst_train_img_path, dst_train_gt_path
                elif op_mode == 'test':
                    src_img_path, src_gt_path = src_test_img_path, src_test_gt_path
                    dst_img_path, dst_gt_path = dst_test_img_path, dst_test_gt_path

                # link img_path
                img_sym_cmd = 'ln "{}"* "{}"'.format(src_img_path, dst_img_path)  # to all files
                subprocess.call(img_sym_cmd, shell=True)
                logger.info(" # Link img files {} -> {}.".format(src_img_path, dst_img_path))

                # link gt_path
                gt_sym_cmd = 'ln "{}"* "{}"'.format(src_gt_path, dst_gt_path)  # to all files
                subprocess.call(gt_sym_cmd, shell=True)
                logger.info(" # Link gt files {} -> {}.".format(src_gt_path, dst_gt_path))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_train(ini, model_dir=None, logger=None):
    cuda_ids = ini['cuda_ids'].split(',')
    train_args = ['--dataset_type', DATASET_TYPE,
                  '--img_path', ini['train_img_path'],
                  '--gt_path', ini['train_gt_path'],
                  '--cuda', ini['cuda'],
                  '--cuda_ids', cuda_ids,
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

    test_args = ['--model_path', ini['pretrain_model_path'],
                 '--dataset_type', ini['dataset_type'],
                 '--img_path', ini['test_img_path'],
                 '--gt_path', ini['test_gt_path'],
                 '--cuda', ini['cuda'],
                 '--rst_path', ini['rst_path']
                 ]

    test.main(test.parse_arguments(test_args), logger=logger)

def main_split_textline(ini, logger=None):
    img_fnames = sorted(utils.get_filenames(ini['dataset_path'], recursive_=True, extensions=utils.IMG_EXTENSIONS))
    ann_fnames = sorted(utils.get_filenames(ini['dataset_path'], recursive_=True, extensions=utils.META_EXTENSION))
    logger.info(" [SPLIT-TEXTLINE] # Total file number to be processed: {:d}.".format(len(img_fnames)))

    for idx, img_fname in enumerate(img_fnames):
        logger.info(" [SPLIT-TEXTLINE] # Processing {} ({:d}/{:d})".format(img_fname, (idx + 1), len(img_fnames)))

        _, img_core_name, img_ext = utils.split_fname(img_fname)
        img = utils.imread(img_fname, color_fmt='RGB')

        # Use CRAFT
        gpu_ = True if ini['cuda'] == 'True' else False
        device = torch.device('cuda' if (torch.cuda.is_available() and gpu_) else 'cpu')
        detector = get_detector(ini['pretrain_model_path'], device, quantize=False)
        easyocr_ini = utils.get_ini_parameters(os.path.join(_this_folder_, ini['ocr_ini_fname']))
        craft_params = load_craft_parameters(easyocr_ini['CRAFT'])

        # Get textbox
        text_boxes = get_textbox(detector, img,
                               canvas_size=craft_params['canvas_size'], mag_ratio=craft_params['mag_ratio'],
                               text_threshold=craft_params['text_threshold'], link_threshold=craft_params['link_threshold'],
                               low_text=craft_params['low_text'], poly=False,
                               device=device, optimal_num_chars=True)

        # Check crop img
        for text_box in text_boxes:
            rect4 = coord.convert_1d_to_matrix(text_box, length=2)
            rect2 = coord.convert_rect4_to_rect2(rect4)
            crop_img = img[rect2[2]:rect2[3], rect2[0]:rect2[1]]
            utils.imshow(crop_img)

        # Load json
        ann_fname = ann_fnames[idx]
        _, ann_core_name, _ = utils.split_fname(ann_fname)
        ann_core_name = ann_core_name.replace('.jpg', '')
        if ann_core_name == img_core_name:
            with open(ann_fname) as json_file:
                json_data = json.load(json_file)
                objects = json_data['objects']
                # pprint.pprint(objects)

        bboxes = []
        texts = []
        for obj in objects:
            class_name = obj['classTitle']
            if class_name != 'textline':
                continue

            [x1, y1], [x2, y2] = obj['points']['exterior']
            text = obj['description']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            rect4 = coord.convert_rect2_to_rect4([x_min, x_max, y_min, y_max])
            bboxes.append(rect4)
            texts.append(text)

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main(args):
    ini = utils.get_ini_parameters(args.ini_fname)
    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == 'GENERATE':
        main_generate(ini['GENERATE'], logger=logger)
    elif args.op_mode == 'SPLIT':
        main_split(ini['SPLIT'], logger=logger)
    elif args.op_mode == 'MERGE':
        main_merge(ini['MERGE'], logger=logger)
    elif args.op_mode == 'TRAIN':
        main_train(ini['TRAIN'], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TEST':
        main_test(ini['TEST'], model_dir=args.model_dir, logger=logger)
    elif args.op_mode == 'TRAIN_TEST':
        ret, model_dir = main_train(ini['TRAIN'], model_dir=args.model_dir, logger=logger)
        main_test(ini['TEST'], model_dir, logger=logger)
        print(" # Trained model directory is {}".format(model_dir))
    elif args.op_mode == 'SPLIT_TEXTLINE':
        main_split_textline(ini['SPLIT_TEXTLINE'], logger=logger)
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", required=True, choices=['TEXTLINE', 'KO', 'MATH'], help="dataset type")
    parser.add_argument("--op_mode", required=True, choices=['GENERATE', 'MERGE', 'SPLIT', 'TRAIN', 'TEST', 'TRAIN_TEST', 'SPLIT_TEXTLINE'], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = 'KO' # TEXTLINE / KO / MATH
OP_MODE = 'TRAIN' # GENERATE / SPLIT / MERGE / TRAIN / TEST / TRAIN_TEST / SPLIT_TEXTLINE
"""
[OP_MODE DESC.]
GENERATE : JSON을 읽어 텍스트라인을 CRAFT 형식으로 변환후 텍스트파일 저장
SPLIT : TRAIN & TEST 비율에 맞춰 각각의 폴더에 저장
MERGE : 만개 단위로 분할된 폴더를 total 폴더로 합침
TRAIN : total/train 폴더 데이터를 이용하여 CRAFT 학습 수행
TEST : total/test 폴더 데이터를 이용하여 CRAFT 평가 수행
SPLIT_TEXTLINE : 각각의 ann 폴더의  데이터를 수식/비수식 영역으로 분리후 각각의 refine_ann 폴더에 JSON 파일 저장  
"""
if DATASET_TYPE != 'TEXTLINE':
    INI_FNAME = _this_basename_ + '_{}'.format(DATASET_TYPE.lower()) + ".ini"
else:
    INI_FNAME = _this_basename_ + ".ini"


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if SELF_TEST_:
            sys.argv.extend(["--dataset_type", DATASET_TYPE])
            sys.argv.extend(["--op_mode", OP_MODE])
            sys.argv.extend(["--ini_fname", INI_FNAME])
            sys.argv.extend(["--logging_"])
            sys.argv.extend(["--console_logging_"])
        else:
            sys.argv.extend(["--help"])

    main(parse_arguments(sys.argv[1:]))