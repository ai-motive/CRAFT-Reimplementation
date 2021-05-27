import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
import general_utils as utils
import file_utils
import coordinates as coord
import train, test
import subprocess
from datetime import datetime
from sklearn.model_selection import train_test_split
from easyocr.detection import get_detector, get_textbox
from easyocr.utils import group_text_box
from str_utils import replace_string_from_dict, get_prev_and_next, is_korean


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


MARGIN = '\t'*20

def str2bool(v):
  return v.lower() in ('yes', 'true', 't', '1')

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

def main_generate(ini, common_info, logger=None):
    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)

    utils.folder_exists(vars['gt_path'], create_=True)

    img_fnames = sorted(utils.get_filenames(vars['img_path'], extensions=utils.IMG_EXTENSIONS))
    ann_fnames = sorted(utils.get_filenames(vars['ann_path'], extensions=utils.META_EXTENSION))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(img_fnames)))

    for idx, img_fname in enumerate(img_fnames):
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

        file_utils.saveResult(img_file=img_core_name, img=img, boxes=bboxes, texts=texts, dirname=vars['gt_path'])
        logger.info(" [GENERATE-OCR] # Generated to {} ({:d}/{:d})".format(vars['gt_path']+img_core_name+'.txt', (idx + 1), len(img_fnames)))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_split(ini, common_info, logger=None):
    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)
        
    utils.folder_exists(vars['img_path'], create_=False)
    utils.folder_exists(vars['gt_path'], create_=False)
    if utils.folder_exists(vars['train_path'], create_=False):
        print(" @ Warning: train dataset path, {}, already exists".format(vars["train_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
        # shutil.rmtree(vars['train_path'])

    if utils.folder_exists(vars['test_path'], create_=False):
        print(" @ Warning: test dataset path, {}, already exists".format(vars["test_path"]))
        ans = input(" % Proceed (y/n) ? ")
        if ans.lower() != 'y':
            sys.exit()
        # shutil.rmtree(vars['test_path'])

    train_ratio = float(vars['train_ratio'])
    test_ratio = round(1.0-train_ratio, 2)

    lower_dataset_type = DATASET_TYPE.lower() if DATASET_TYPE != 'TEXTLINE' else ''
    tgt_dir = 'craft_{}_gt'.format(lower_dataset_type)
    gt_list = sorted(utils.get_filenames(vars['gt_path'], extensions=utils.TEXT_EXTENSIONS))
    train_gt_list, test_gt_list = train_test_split(gt_list, train_size=train_ratio, random_state=2000)
    train_img_list, test_img_list = [gt_path.replace(tgt_dir, 'img').replace('.txt', '.jpg').replace('gt_', '') for gt_path in train_gt_list], \
                                    [gt_path.replace(tgt_dir, 'img').replace('.txt', '.jpg').replace('gt_', '') for gt_path in test_gt_list]

    train_img_path, test_img_path = os.path.join(vars['train_path'], 'img/'), os.path.join(vars['test_path'], 'img/')
    train_gt_path, test_gt_path = os.path.join(vars['train_path'], tgt_dir+'/'), os.path.join(vars['test_path'], tgt_dir+'/')
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
            logger.info(" # Link gt files {}\n{}->{}.".format(gt_list[0], MARGIN, gt_link_path))

            # link img_path
            for img_path in img_list:
                img_sym_cmd = 'ln "{}" "{}"'.format(img_path, img_link_path)  # to all files
                subprocess.call(img_sym_cmd, shell=True)
            logger.info(" # Link img files {}\n{}->{}.".format(img_list[0], MARGIN, img_link_path))

    print(" # (train, test) = ({:d}, {:d}) -> {:d} % ".
          format(len(train_gt_list), len(test_gt_list), int(float(len(train_gt_list))/float(len(train_gt_list)+len(test_gt_list))*100)))
    return True


def main_merge(ini, common_info, logger=None):
    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)
        
    utils.folder_exists(vars['total_dataset_path'], create_=True)

    datasets = [dataset for dataset in os.listdir(vars['dataset_path']) if dataset != 'total']
    sort_datasets = sorted(datasets, key=lambda x: (int(x.split('_')[0])))

    lower_dataset_type = DATASET_TYPE.lower() if DATASET_TYPE != 'TEXTLINE' else ''
    tgt_dir = 'craft_{}_gt'.format(lower_dataset_type)
    if len(sort_datasets) != 0:
        for dir_name in sort_datasets:
            src_train_path, src_test_path = os.path.join(vars['dataset_path'], dir_name, 'train'), os.path.join(vars['dataset_path'], dir_name, 'test')
            src_train_img_path, src_train_gt_path = os.path.join(src_train_path, 'img/'), os.path.join(src_train_path, tgt_dir+'/')
            src_test_img_path, src_test_gt_path = os.path.join(src_test_path, 'img/'), os.path.join(src_test_path, tgt_dir+'/')

            dst_train_path, dst_test_path = os.path.join(vars['total_dataset_path'], 'train'), os.path.join(vars['total_dataset_path'], 'test')
            dst_train_img_path, dst_train_gt_path = os.path.join(dst_train_path, 'img/'), os.path.join(dst_train_path, tgt_dir+'/')
            dst_test_img_path, dst_test_gt_path = os.path.join(dst_test_path, 'img/'), os.path.join(dst_test_path, tgt_dir+'/')

            if utils.folder_exists(dst_train_img_path) and utils.folder_exists(dst_train_gt_path) and \
                    utils.folder_exists(dst_test_img_path) and utils.folder_exists(dst_test_gt_path):
                logger.info(" # Already {} is exist".format(vars['total_dataset_path']))
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
                logger.info(" # Link img files {}\n{}->{}.".format(src_img_path, MARGIN, dst_img_path))

                # link gt_path
                gt_sym_cmd = 'ln "{}"* "{}"'.format(src_gt_path, dst_gt_path)  # to all files
                subprocess.call(gt_sym_cmd, shell=True)
                logger.info(" # Link gt files {}\n{}->{}.".format(src_gt_path, MARGIN, dst_gt_path))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def main_train(ini, common_info, logger=None):
    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)
        
    cuda_ids = vars['cuda_ids'].split(',')
    latest_model_dir = utils.get_model_dir(root_dir=vars['root_model_path'], model_file=vars['model_name'], version='latest')
    latest_model_path = os.path.join(latest_model_dir, vars['model_name'])

    train_args = [
        '--tgt_class', common_info['tgt_class'].upper(),
        '--img_path', vars['train_img_path'],
        '--gt_path', vars['train_gt_path'],
        '--pretrain_model_path', latest_model_path,
        '--cuda', vars['cuda'],
        '--cuda_ids', cuda_ids,
        '--resume', vars['resume'],
        '--valid_epoch', vars['valid_epoch'],
        '--batch_size', vars['batch_size'],
        '--learning_rate', vars['learning_rate'],
        '--momentum', vars['momentum'],
        '--weight_decay', vars['weight_decay'],
        '--gamma', vars['gamma'],
        '--num_workers', vars['num_workers'],
    ]

    train.main(train.parse_arguments(train_args), logger=logger)

    return True

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

def main_split_textline(ini, common_info, logger=None):
    # Init. path variables
    global box_color, rst_dir_name
    except_dir_names = common_info['except_dir_names'].replace(' ', '').split(',')
    vars = {}
    for key, val in ini.items():
        vars[key] = replace_string_from_dict(val, common_info)
    img_mode = vars['img_mode']
    link_, border_, save_detect_box_img_, save_refine_box_img_ = \
        str2bool(vars['link_']), str2bool(vars['border_']), str2bool(vars['save_detect_box_img_']), str2bool(vars['save_refine_box_img_'])

    # Init. CRAFT
    gpu_ = str2bool(vars['cuda'])
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu_) else 'cpu')

    ko_model_dir, math_model_dir = utils.get_model_dir(root_dir=vars['ko_model_path'], model_file=vars['ko_model_name']), \
                                        utils.get_model_dir(root_dir=vars['math_model_path'], model_file=vars['math_model_name'])
    ko_detector = get_detector(os.path.join(ko_model_dir, vars['ko_model_name']), device, quantize=False)
    math_detector = get_detector(os.path.join(math_model_dir, vars['math_model_name']), device, quantize=False)

    easyocr_ini = utils.get_ini_parameters(os.path.join(_this_folder_, vars['ocr_ini_fname']))
    craft_params = load_craft_parameters(easyocr_ini['CRAFT'])

    datasets = [dataset for dataset in os.listdir(vars['textline_dataset_path']) if dataset != 'total']
    sort_datasets = sorted(datasets, key=lambda x: (int(x.split('_')[0])))

    # Preprocess datasets
    if link_:
        link_datasets(src_dir_path=vars['textline_dataset_path'], dst_dir_path=vars['refine_dataset_path'],
                      dir_names=sort_datasets, except_dir_names=except_dir_names, tgt_dir_name='img/', logger=logger)

    # Load and split textlines
    for dir_name in sort_datasets:
        if dir_name in except_dir_names:
            logger.info(" # {} has already been split. ".format(dir_name))
            continue

        img_path = os.path.join(vars['textline_dataset_path'], dir_name, 'img/')
        ann_path = os.path.join(vars['textline_dataset_path'], dir_name, 'ann/')

        img_fnames = sorted(utils.get_filenames(img_path, extensions=utils.IMG_EXTENSIONS))
        ann_fnames = sorted(utils.get_filenames(ann_path, extensions=utils.META_EXTENSION))

        logger.info(" [SPLIT-TEXTLINE] # Total file number to be processed: {:d}.".format(len(img_fnames)))

        for idx, img_fname in enumerate(img_fnames):
            logger.info(" [SPLIT-TEXTLINE] # Processing {} ({:d}/{:d})".format(img_fname, (idx+1), len(img_fnames)))
            img = utils.imread(os.path.join(img_fname), color_fmt='RGB')
            draw_detect_img, draw_refine_img = img.copy(), img.copy()

            # Load json
            img_bname = os.path.basename(img_fname)
            ann_fname = ann_fnames[idx]
            ann_bname = os.path.basename(ann_fname)
            if ann_bname.replace('.json', '') == img_bname:
                with open(os.path.join(ann_fname)) as json_file:
                    json_data = json.load(json_file)
                    objects = json_data['objects']
                    # pprint.pprint(objects)

            # Get ground truths
            gts = []
            gt_crop_boxes, gt_crop_imgs = [], []
            for idx, obj in reversed(list(enumerate(objects))):
                class_name = obj['classTitle']
                if (class_name != common_info['tgt_class']):
                    continue

                [x1, y1], [x2, y2] = obj['points']['exterior']
                text = obj['description']

                # Remove textline objects
                del objects[idx]

                x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
                if x_max - x_min <= 0 or y_max - y_min <= 0:
                    continue

                rect2 = [x_min, x_max, y_min, y_max]
                rect4 = coord.convert_rect2_to_rect4(rect2)
                gts.append([rect4, text, class_name])
                gt_crop_boxes.append(rect2)
                gt_crop_imgs.append(img[y_min:y_max, x_min:x_max])

                box = [x_min, y_min, x_max, y_max]
                draw_detect_img = utils.draw_box_on_img(draw_detect_img, box, color=utils.BLUE)
                draw_refine_img = utils.draw_box_on_img(draw_refine_img, box, color=utils.BLUE)

            # Get predict results
            predicts = []
            for detector in [ko_detector, math_detector]:
                if img_mode == 'normal':
                    boxes = [[-1, -1, -1, -1]]
                    imgs = [img]

                elif img_mode == 'crop':
                    boxes = gt_crop_boxes
                    imgs = gt_crop_imgs

                for input_box, input_img in zip(boxes, imgs):
                    tgt_class = 'ko' if (detector is ko_detector) else ('math' if (detector is math_detector) else 'None')

                    # # Make border
                    border_margin = 0
                    if border_:
                        border_color = utils.WHITE
                        border_margin = 30
                        input_img = cv2.copyMakeBorder(input_img,
                                                      border_margin, border_margin, border_margin, border_margin,
                                                      cv2.BORDER_CONSTANT, value=border_color)

                    boxes = get_textbox(detector, input_img,
                                        canvas_size=craft_params['canvas_size'], mag_ratio=craft_params['mag_ratio'],
                                        text_threshold=craft_params['text_threshold'], link_threshold=craft_params['link_threshold'],
                                        low_text=craft_params['low_text'], poly=False,
                                        device=device, optimal_num_chars=True)

                    if border_:
                        boxes = [np.array([box[0]-border_margin, box[1]-border_margin, box[2]-border_margin, box[3]-border_margin,
                                            box[4]-border_margin, box[5]-border_margin, box[6]-border_margin, box[7]-border_margin])  for box in boxes]

                    horizontal_list, _ = group_text_box(boxes, craft_params['slope_ths'],
                                                        craft_params['ycenter_ths'], craft_params['height_ths'],
                                                        craft_params['width_ths'], craft_params['add_margin'])

                    for h_box in horizontal_list:
                        if input_box[0] == -1:
                            new_h_box = h_box
                        else:
                            new_h_box = coord.calc_global_box_pos_in_box(input_box, h_box)

                        x_min, x_max, y_min, y_max = new_h_box
                        rect4 = coord.convert_rect2_to_rect4([x_min, x_max, y_min, y_max])
                        predicts.append([rect4, '', tgt_class])
                        # crop_img = img[y_min:y_max, x_min:x_max]
                        # utils.imshow(crop_img)

                        box = (x_min, y_min, x_max, y_max)

                        if tgt_class == 'ko':
                            box_color = utils.BROWN
                        if tgt_class == 'math':
                            box_color = utils.MAGENTA

                        draw_detect_img = utils.draw_box_on_img(draw_detect_img, box, color=box_color)

            # Save result image
            ko_model_epoch, math_model_epoch = vars['ko_model_name'].split('_')[-1].replace('.pth', ''), \
                                               vars['math_model_name'].split('_')[-1].replace('.pth', '')
            rst_dir_name = 'ko_' + ko_model_epoch + '_' + 'math_' + math_model_epoch
            rst_dir_path = os.path.join(vars['rst_path'], rst_dir_name, 'draw_box')
            if save_detect_box_img_:
                utils.folder_exists(rst_dir_path, create_=True)
                utils.imwrite(draw_detect_img, os.path.join(rst_dir_path, f'[{img_mode}] ' + img_bname))

            # Compare GT. & PRED.
            refine_gts = refine_ground_truths_by_predict_values(gts, predicts) # test input : GTS, PREDS

            # Draw refined boxes
            if save_refine_box_img_:
                for rf_box, rf_text, rf_class in refine_gts:
                    rf_rect2 = coord.convert_rect4_to_rect2(rf_box)
                    x_min, x_max, y_min, y_max = rf_rect2
                    box = (x_min, y_min, x_max, y_max)

                    if rf_class == 'ko':
                        box_color = utils.BROWN
                    if rf_class == 'math':
                        box_color = utils.MAGENTA

                    draw_refine_img = utils.draw_box_on_img(draw_refine_img, box, color=box_color, thickness=3)

                rst_dir_path = os.path.join(vars['rst_path'], rst_dir_name, 'refine_box')
                utils.folder_exists(rst_dir_path, create_=True)
                utils.imwrite(draw_refine_img, os.path.join(rst_dir_path, f'[{img_mode}] ' + img_bname))

            # # Insert refine_gts to json
            # obj_id = objects[-1]['id'] + 1
            # refine_json_data, refine_obj_id = update_json_from_results(json_data, obj_id,
            #                                                            ['ko', 'math'], refine_gts)
            #
            # # Save refined json
            # rst_ann_fname = ann_fname.replace(vars['textline_dataset_path'], vars['refine_dataset_path'])
            # with open(rst_ann_fname, 'w', encoding='utf-8') as f:
            #     json.dump(refine_json_data, f, ensure_ascii=False, indent=4)

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True

def link_datasets(src_dir_path, dst_dir_path, dir_names, except_dir_names=None, tgt_dir_name='img/', logger=None):
    if dir_names:
        for dir_name in dir_names:
            if dir_name in except_dir_names:
                logger.info(" # {} has already been split. ".format(dir_name))
                continue

            src_path = os.path.join(src_dir_path, dir_name, tgt_dir_name)

            dst_path = os.path.join(dst_dir_path, dir_name, tgt_dir_name)

            if utils.folder_exists(dst_path):
                logger.info(" # Already {} is exist".format(dst_path))
            else:
                utils.folder_exists(dst_path, create_=True)

            # check & link img_path, ann_path
            src_fnames = sorted(utils.get_filenames(src_path, extensions=utils.IMG_EXTENSIONS))
            src_bnames = [os.path.basename(src_fname) for src_fname in src_fnames]
            dst_fnames = sorted(utils.get_filenames(dst_path, extensions=utils.IMG_EXTENSIONS))
            dst_bnames = [os.path.basename(dst_fname) for dst_fname in dst_fnames]

            if any(src_bname not in dst_bnames for src_bname in src_bnames):
                img_sym_cmd = 'ln "{}"* "{}"'.format(src_path, dst_path)  # to all files
                subprocess.call(img_sym_cmd, shell=True)
                logger.info(" # Link img files {}\n{}->\t{}.".format(src_path, MARGIN, dst_path))
    else:
        logger.info(" [SPLIT-TEXTLINE] # Sorted dataset is empty !!!")

def refine_ground_truths_by_predict_values(gts, preds):
    refine_gts = []
    for gt_idx, (gt_box, gt_text, gt_class) in enumerate(gts):
        gt_rect2 = coord.convert_rect4_to_rect2(gt_box) # [min_x, max_x, min_y, max_y]
        gt_min_x, gt_max_x, gt_min_y, gt_max_y = gt_rect2

        # 중심점으로 gt 내부에 있는 pred. 후보 영역 추출
        cand_preds = []
        for pred in preds:
            pred_box, pred_text, pred_class = pred
            pred_rect2 = coord.convert_rect4_to_rect2(pred_box)
            pred_min_x, pred_max_x, pred_min_y, pred_max_y = pred_rect2
            pred_center_x, pred_center_y = (pred_min_x+pred_max_x)/2, (pred_min_y+pred_max_y)/2

            if (gt_min_x < pred_center_x < gt_max_x) and (gt_min_y < pred_center_y < gt_max_y):
                cand_preds.append(pred)

        # (x, y) 좌표를 기반으로 sorting
        sort_preds = sorted(cand_preds, key=lambda x: (x[0][0], x[0][1]))

        # 박스 좌표 및 사이즈로 중복 or 포함된 preds. 박스 제거
        for i, sort_pred in reversed(list(enumerate(sort_preds))):
            sort_pred_box, sort_pred_text, sort_pred_class = sort_pred
            sort_pred_rect2 = coord.convert_rect4_to_rect2(sort_pred_box)
            sort_min_x, sort_max_x, sort_min_y, sort_max_y = sort_pred_rect2
            sort_center_x, sort_center_y = (sort_min_x + sort_max_x) / 2, (sort_min_y + sort_max_y) / 2
            sort_area_size = (sort_max_x-sort_min_x)*(sort_max_y-sort_min_y)
            if len(sort_preds) > 1:
                for j, ref_pred in reversed(list(enumerate(sort_preds[:i]))):
                    ref_pred_box, ref_pred_text, ref_pred_class = ref_pred
                    ref_pred_rect2 = coord.convert_rect4_to_rect2(ref_pred_box)
                    ref_min_x, ref_max_x, ref_min_y, ref_max_y = ref_pred_rect2
                    ref_area_size = (ref_max_x - ref_min_x) * (ref_max_y - ref_min_y)

                    # 두 박스의 넓이가 90% 이상 일치하거나 박스 4점이 모두 포함되면 제거
                    if ((sort_area_size / ref_area_size) >= 0.9) and (abs(sort_min_x-ref_min_x) <= 10) or \
                            ((ref_min_x < sort_min_x < ref_max_x) and (ref_min_x < sort_max_x < ref_max_x) and
                                (ref_min_y < sort_min_y < ref_max_y) and (ref_min_y < sort_max_y < ref_max_y)):
                        del sort_preds[i]
                        break

        remove_preds = sort_preds

        split_gts = []
        # 예측 값이 없는 경우
        if len(remove_preds) == 0:
            min_x, min_y = gt_min_x, gt_min_y
            max_x, max_y = gt_max_x, gt_max_y
            rect4 = coord.convert_rect2_to_rect4([min_x, max_x, min_y, max_y])
            split_gts.append([rect4, '', 'math'])
        else:
            # Create refined gts
            for k, remove_pred in enumerate(remove_preds):
                remove_pred_box, remove_pred_text, remove_pred_class = remove_pred
                remove_pred_rect2 = coord.convert_rect4_to_rect2(remove_pred_box)
                remove_min_x, remove_max_x, remove_min_y, remove_max_y = remove_pred_rect2

                # 예측 개수를 기반으로 x, y값 조정
                if len(remove_preds) == 1:
                    min_x, min_y = gt_min_x, gt_min_y
                    max_x, max_y = gt_max_x, gt_max_y
                    rect4 = coord.convert_rect2_to_rect4([min_x, max_x, min_y, max_y])
                    split_gts.append([rect4, '', remove_pred_class])

                # 예측 개수가 2개 이상 일때
                else:
                    # 첫번째 영역 처리
                    if k == 0:
                        min_x, min_y = gt_min_x, gt_min_y
                        max_x, max_y = remove_max_x, gt_max_y
                        rect4 = coord.convert_rect2_to_rect4([min_x, max_x, min_y, max_y])
                        split_gts.append([rect4, '', remove_pred_class])

                    # 마지막 영역 처리
                    elif k == len(remove_preds)-1:
                        min_x, min_y = remove_min_x, gt_min_y
                        max_x, max_y = gt_max_x, gt_max_y
                        rect4 = coord.convert_rect2_to_rect4([min_x, max_x, min_y, max_y])
                        split_gts.append([rect4, '', remove_pred_class])

                    # 중간 영역 처리
                    else:
                        min_x, min_y = remove_min_x, gt_min_y
                        max_x, max_y = remove_max_x, gt_max_y
                        rect4 = coord.convert_rect2_to_rect4([min_x, max_x, min_y, max_y])
                        split_gts.append([rect4, '', remove_pred_class])

        # pred_class를 기반으로 text filling
        ch_pos = 0
        for l, split_gt in enumerate(split_gts):
            split_gt_box, split_gt_text, split_gt_class = split_gt
            refine_gts.append([split_gt_box, '', split_gt_class])
            # gt_text = gt_text[ch_pos:]
            for m, (prev_ch, curr_ch, next_ch) in enumerate(get_prev_and_next(gt_text[ch_pos:])):
                if (len(gt_text) <= 1) and (curr_ch == ' ' or curr_ch == ''):
                    refine_gts[-1][1] += curr_ch
                    ch_pos += 1
                    # refine_gts[-1][2] = 'ko'
                    break

                # 첫번째 문자 처리
                if ch_pos == 0:
                    if is_korean(curr_ch):
                        refine_gts[-1][1] += curr_ch
                        ch_pos += 1
                        # refine_gts[-1][2] = 'ko'
                    elif is_korean(curr_ch) == False:
                        refine_gts[-1][1] += curr_ch
                        ch_pos += 1
                        # refine_gts[-1][2] = 'math'
                else:
                    # (한글+빈칸) and (prev_ch_class == curr_ch_class)
                    if (is_korean(prev_ch) and (curr_ch == ' ')) or ((prev_ch == ' ' or prev_ch == None) and is_korean(curr_ch)) \
                            or (is_korean(prev_ch) and is_korean(curr_ch)):
                        refine_gts[-1][1] += curr_ch
                        ch_pos += 1
                        # refine_gts[-1][2] = 'ko'

                    # (수식+빈칸) and (prev_ch_class == curr_ch_class)
                    elif (not(is_korean(prev_ch)) and (curr_ch == ' ')) or ((prev_ch == ' ' or prev_ch == None) and not(is_korean(curr_ch)))  \
                            or ((is_korean(prev_ch) == False) and (is_korean(curr_ch) == False)):
                        refine_gts[-1][1] += curr_ch
                        ch_pos += 1
                        # refine_gts[-1][2] = 'math'

                    # class가 바뀔때
                    curr_class = refine_gts[-1][2]
                    if (curr_class == 'ko' and not(is_korean(next_ch)) and (next_ch != ' ')) or \
                            (curr_class == 'math' and (is_korean(next_ch)) and (next_ch != ' ')) or \
                                next_ch == None:
                        break

    return refine_gts

def update_json_from_results(json_data, obj_id, class_names, results):
    for i, (box, value, class_name) in enumerate(results):
        if class_name in class_names:
            rect2 = coord.convert_rect4_to_rect2(box)
            update_obj = get_obj_data(obj_id, class_name, rect2, value)
            json_data['objects'].append(update_obj)
            obj_id += 1

    return json_data, obj_id

def get_obj_data(obj_id, classTitle, box, value):
    if classTitle == 'table':
        classId = 2790491
        value = ''
    elif classTitle == 'graph':
        classId = 2772037
        value = ''
    elif classTitle == 'textline':
        classId = 2772036
    elif classTitle == 'math':
        classId = 2883527
    elif classTitle == 'ko':
        classId = 2883530

    date = datetime.today().strftime("%Y%m%d%H%M%S")
    year, month, day, hour, minute, second = date[:4], date[4:6], date[6:8], date[8:10], date[10:12], date[12:14]

    obj_data = {}
    update_obj = update_obj_data(obj_data,
                                 id=obj_id, classId=classId,
                                 description=value, geometryType='rectangle',
                                 labelerLogin='freewheelin',
                                 createdAt=f'{year}-{month}-{day}T{hour}:{minute}:{second}.271Z', updatedAt=f'{year}-{month}-{day}T{hour}:{minute}:{second}.271Z',
                                 classTitle=classTitle,
                                 tags=[],
                                 points={
                                     'exterior': [[box[0], box[2]],
                                                  [box[1], box[3]]],
                                     'interior': [[]],
                                 })
    return update_obj

def update_obj_data(obj_data, id, classId, description, geometryType, labelerLogin, createdAt, updatedAt, tags, classTitle, points):
    obj_data['id'] = id
    obj_data['classId'] = classId
    obj_data['description'] = description
    obj_data['geometryType'] = geometryType
    obj_data['labelerLogin'] = labelerLogin
    obj_data['createdAt'] = createdAt
    obj_data['updatedAt'] = updatedAt
    obj_data['tags'] = tags
    obj_data['classTitle'] = classTitle
    obj_data['classId'] = classId
    obj_data['points'] = points
    return obj_data

def main(args):
    ini = utils.get_ini_parameters(args.ini_fname)
    common_info = {}
    for key, val in ini['COMMON'].items():
        common_info[key] = val

    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == 'GENERATE':
        main_generate(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'SPLIT':
        main_split(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'MERGE':
        main_merge(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'TRAIN':
        main_train(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'TEST':
        main_test(ini[args.op_mode], common_info, logger=logger)
    elif args.op_mode == 'TRAIN_TEST':
        ret, model_dir = main_train(ini['TRAIN'], common_info, logger=logger)
        main_test(ini['TEST'], common_info, logger=logger)
        print(" # Trained model directory is {}".format(model_dir))
    elif args.op_mode == 'SPLIT_TEXTLINE':
        main_split_textline(ini[args.op_mode], common_info, logger=logger)
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
DATASET_TYPE = 'TEXTLINE' # KO / MATH / TEXTLINE
OP_MODE = 'SPLIT_TEXTLINE' # GENERATE / SPLIT / MERGE / TRAIN / TEST / TRAIN_TEST / SPLIT_TEXTLINE
"""
[OP_MODE DESC.]
GENERATE       : JSON을 읽어 텍스트라인을 CRAFT 형식으로 변환후 텍스트파일 저장
SPLIT          : TRAIN & TEST 비율에 맞춰 각각의 폴더에 저장
MERGE          : 만개 단위로 분할된 폴더를 total 폴더로 합침
TRAIN          : total/train 폴더 데이터를 이용하여 CRAFT 학습 수행
TEST           : total/test 폴더 데이터를 이용하여 CRAFT 평가 수행
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
