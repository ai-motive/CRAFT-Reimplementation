import os
import sys
import json
import argparse
import shutil
import numpy as np
import torch
import cv2
import file_utils
import train, test
import subprocess
from datetime import datetime
from sklearn.model_selection import train_test_split
from PIL import ImageDraw, Image
from easyocr.detection import get_detector, get_textbox
from easyocr.utils import group_text_box
from python_utils.common import general as cg, logger as cl, string as cs
from python_utils.image import general as ig, process as ip, coordinates as ic, object
from python_utils.json import general as jg
import supervisely_lib as sly


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]

MARGIN = '\t' * 20

# DATASET_TYPE
KO, MATH, KO_MATH, TEXTLINE = 'KO', 'MATH', 'KO_MATH', 'TEXTLINE'

# OP_MODE
PREPROCESS_ALL, GENERATE, SPLIT, MERGE, TRAIN, TEST, TRAIN_TEST, SPLIT_TEXTLINE = \
    'PREPROCESS_ALL', 'GENERATE', 'SPLIT', 'MERGE', 'TRAIN', 'TEST', 'TRAIN_TEST', 'SPLIT_TEXTLINE'

LINK, COPY = 'LINK', 'COPY'

# ANN CLASSES (TABLE / GRAPH / TEXTLINE, KO, MATH)
TABLE, GRAPH = 'TABLE', 'GRAPH'


def load_craft_parameters(ini):
    params = {
        'min_size': int(ini['min_size']), 'text_threshold': float(ini['text_threshold']),
        'low_text': float(ini['low_text']), 'link_threshold': float(ini['link_threshold']),
        'canvas_size': int(ini['canvas_size']), 'mag_ratio': float(ini['mag_ratio']),
        'slope_ths': float(ini['slope_ths']), 'ycenter_ths': float(ini['ycenter_ths']),
        'height_ths': float(ini['height_ths']), 'width_ths': float(ini['width_ths']),
        'add_margin': float(ini['add_margin'])
    }
    return params


def main_generate(ini, common_info, logger=None):
    # Init. local variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(vars['gt_path'], create_=True)

    img_fnames = sorted(cg.get_filenames(vars['img_path'], extensions=ig.IMG_EXTENSIONS))
    ann_fnames = sorted(cg.get_filenames(vars['ann_path'], extensions=jg.META_EXTENSION))
    logger.info(" [GENERATE] # Total file number to be processed: {:d}.".format(len(img_fnames)))

    for idx, img_fname in enumerate(img_fnames):
        _, img_core_name, img_ext = cg.split_fname(img_fname)
        img = ig.imread(img_fname, color_fmt='RGB')

        # Load json
        ann_fname = ann_fnames[idx]
        _, ann_core_name, _ = cg.split_fname(ann_fname)
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
            if class_name != common_info['dataset_type'].lower():
                continue

            [x1, y1], [x2, y2] = obj['points']['exterior']
            text = obj['description']
            x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            rect4 = ic.convert_rect2_to_rect4([x_min, x_max, y_min, y_max])
            bboxes.append(rect4)
            texts.append(text)

        file_utils.saveResult(img_file=img_core_name, img=img, boxes=bboxes, texts=texts, dirname=vars['gt_path'])
        logger.info(
            " [GENERATE-OCR] # Generated to {} ({:d}/{:d})".format(vars['gt_path'] + img_core_name + '.txt', (idx + 1),
                                                                   len(img_fnames)))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, GENERATE))
    return True


def main_split(ini, common_info, logger=None):
    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(vars['img_path'], create_=False)
    cg.folder_exists(vars['gt_path'], create_=False)
    cg.folder_exists(vars['train_path'], create_=False)
    cg.folder_exists(vars['test_path'], create_=False)

    train_ratio = float(vars['train_ratio'])
    test_ratio = round(1.0 - train_ratio, 2)

    lower_dataset_type = common_info['dataset_type'].lower() if common_info['dataset_type'] != TEXTLINE else ''
    if lower_dataset_type:
        tgt_dir = 'craft_{}_gt'.format(lower_dataset_type)
    else:
        tgt_dir = 'craft_gt'
    gt_list = sorted(cg.get_filenames(vars['gt_path'], extensions=cg.TEXT_EXTENSIONS))
    train_gt_list, test_gt_list = train_test_split(gt_list, train_size=train_ratio, random_state=2000)
    train_img_list, test_img_list = [gt_path.replace(tgt_dir, 'img').replace('.txt', '.jpg').replace('gt_', '') for
                                     gt_path in train_gt_list], \
                                    [gt_path.replace(tgt_dir, 'img').replace('.txt', '.jpg').replace('gt_', '') for
                                     gt_path in test_gt_list]

    train_img_path, test_img_path = os.path.join(vars['train_path'], 'img/'), os.path.join(vars['test_path'], 'img/')
    train_gt_path, test_gt_path = os.path.join(vars['train_path'], tgt_dir + '/'), os.path.join(vars['test_path'], tgt_dir + '/')
    cg.folder_exists(train_img_path, create_=True), cg.folder_exists(test_img_path, create_=True)
    cg.folder_exists(train_gt_path, create_=True), cg.folder_exists(test_gt_path, create_=True)

    # Apply symbolic link for gt & img path
    if len(gt_list) != 0:
        for op_mode in [TRAIN, TEST]:
            if op_mode == TRAIN:
                gt_list = train_gt_list
                img_list = train_img_list

                gt_link_path = train_gt_path
                img_link_path = train_img_path
            elif op_mode == TEST:
                gt_list = test_gt_list
                img_list = test_img_list

                gt_link_path = test_gt_path
                img_link_path = test_img_path

            # link gt files
            for gt_path in gt_list:
                gt_sym_cmd = 'ln "{}" "{}"'.format(gt_path, gt_link_path)  # to all files
                subprocess.call(gt_sym_cmd, shell=True)
            logger.info(" # Link gt files {}\n{}->{}.".format(gt_list[0], MARGIN, gt_link_path))

            # link img files
            for img_path in img_list:
                img_sym_cmd = 'ln "{}" "{}"'.format(img_path, img_link_path)  # to all files
                subprocess.call(img_sym_cmd, shell=True)
            logger.info(" # Link img files {}\n{}->{}.".format(img_list[0], MARGIN, img_link_path))

    print(" # (train, test) = ({:d}, {:d}) -> {:d} % ".
          format(len(train_gt_list), len(test_gt_list),
                 int(float(len(train_gt_list)) / float(len(train_gt_list) + len(test_gt_list)) * 100)))
    return True


def main_merge(ini, common_info, logger=None):
    # Init. path variables
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cg.folder_exists(vars['total_dataset_path'], create_=True)

    datasets = [dataset for dataset in os.listdir(vars['dataset_path']) if (dataset != 'total') and ('meta.json' not in dataset)]
    sort_datasets = sorted(datasets, key=lambda x: (int(x.split('_')[0])))

    lower_dataset_type = common_info['dataset_type'].lower() if common_info['dataset_type'] != TEXTLINE else ''
    if lower_dataset_type:
        tgt_dir = 'craft_{}_gt'.format(lower_dataset_type)
    else:
        tgt_dir = 'craft_gt'
    if len(sort_datasets) != 0:
        for dir_name in sort_datasets:
            src_train_path, src_test_path = os.path.join(vars['dataset_path'], dir_name, TRAIN.lower()), \
                                                os.path.join(vars['dataset_path'], dir_name, TEST.lower())
            src_train_img_path, src_train_gt_path = os.path.join(src_train_path, 'img/'), \
                                                        os.path.join(src_train_path, tgt_dir + '/')
            src_test_img_path, src_test_gt_path = os.path.join(src_test_path, 'img/'), \
                                                    os.path.join(src_test_path, tgt_dir + '/')

            dst_train_path, dst_test_path = os.path.join(vars['total_dataset_path'], TRAIN.lower()), \
                                                os.path.join(vars['total_dataset_path'], TEST.lower())
            dst_train_img_path, dst_train_gt_path = os.path.join(dst_train_path, 'img/'), \
                                                        os.path.join(dst_train_path, tgt_dir + '/')
            dst_test_img_path, dst_test_gt_path = os.path.join(dst_test_path, 'img/'), \
                                                    os.path.join(dst_test_path, tgt_dir + '/')

            if cg.folder_exists(dst_train_img_path) and cg.folder_exists(dst_train_gt_path) and \
                    cg.folder_exists(dst_test_img_path) and cg.folder_exists(dst_test_gt_path):
                logger.info(" # Already {} is exist".format(vars['total_dataset_path']))
            else:
                cg.folder_exists(dst_train_img_path, create_=True), cg.folder_exists(dst_train_gt_path, create_=True)
                cg.folder_exists(dst_test_img_path, create_=True), cg.folder_exists(dst_test_gt_path, create_=True)

            # Apply symbolic link for gt & img path
            for op_mode in [TRAIN, TEST]:
                if op_mode == TRAIN:
                    src_img_path, src_gt_path = src_train_img_path, src_train_gt_path
                    dst_img_path, dst_gt_path = dst_train_img_path, dst_train_gt_path
                elif op_mode == TEST:
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
        vars[key] = cs.replace_string_from_dict(val, common_info)

    cuda_ids = vars['cuda_ids'].split(',')
    latest_model_dir = cg.get_model_dir(root_dir=vars['root_model_path'], model_file=vars['model_name'],
                                        version='latest')
    latest_model_path = os.path.join(latest_model_dir, vars['model_name'])

    train_args = [
        '--tgt_class', common_info['tgt_class'],
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
        model_dir = max([os.path.join(ini['model_root_path'], d) for d in os.listdir(ini["model_root_path"])],
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
    vars = {}
    for key, val in ini.items():
        vars[key] = cs.replace_string_from_dict(val, common_info)

    except_dir_names = vars['except_dir_names'].replace(' ', '').split(',')
    img_mode = vars['img_mode']
    link_, copy_, border_, save_detect_box_img_, save_refine_box_img_ = \
        cs.string_to_boolean(vars['link_']), cs.string_to_boolean(vars['copy_']), cs.string_to_boolean(vars['border_']), \
        cs.string_to_boolean(vars['save_detect_box_img_']), cs.string_to_boolean(vars['save_refine_box_img_'])

    # Init. CRAFT
    gpu_ = cs.string_to_boolean(vars['cuda'])
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu_) else 'cpu')

    ko_model_dir, math_model_dir = cg.get_model_dir(root_dir=vars['ko_model_path'], model_file=vars['ko_model_name']), \
                                        cg.get_model_dir(root_dir=vars['math_model_path'], model_file=vars['math_model_name'])
    ko_detector = get_detector(os.path.join(ko_model_dir, vars['ko_model_name']), device, quantize=False)
    math_detector = get_detector(os.path.join(math_model_dir, vars['math_model_name']), device, quantize=False)

    easyocr_ini = cg.get_ini_parameters(os.path.join(_this_folder_, vars['ocr_ini_fname']))
    craft_params = load_craft_parameters(easyocr_ini['CRAFT'])

    project = sly.Project(directory=vars['textline_dataset_path'], mode=sly.OpenMode.READ)

    # Preprocess datasets
    if link_:
        link_or_copy_datasets(src_dir_path=vars['textline_dataset_path'], dst_dir_path=vars['refine_dataset_path'],
                              dir_names=project.datasets.keys(), except_dir_names=except_dir_names,
                              tgt_dir_name='img/', mode=LINK, logger=logger)

    if copy_:
        link_or_copy_datasets(src_dir_path=vars['textline_dataset_path'], dst_dir_path=vars['refine_dataset_path'],
                              dir_names=project.datasets.keys(), except_dir_names=except_dir_names,
                              tgt_dir_name='ann/', mode=COPY, logger=logger)

    # Load and split textlines
    for dataset in project:
        sly.logger.info('Processing dataset: {}/{}'.format(project.name, dataset.name))

        if dataset.name in except_dir_names:
            logger.info(" # {} has already been split. ".format(dataset.name))
            continue

        for item_idx, item_name in enumerate(dataset):
            item_paths = dataset.get_item_paths(item_name)
            ann, json_data = sly.Annotation.load_json_file(item_paths.ann_path, project.meta)


            raw_img = sly.image.read(item_paths.img_path)
            draw_detect_img, draw_refine_img = raw_img.copy(), raw_img.copy()

            # Draw textline box
            for label in ann.labels:
                if label.obj_class.name == TEXTLINE.lower():
                    label.geometry.draw_contour(draw_detect_img, color=label.obj_class.color, config=label.obj_class.geometry_config, thickness=3)
                    label.geometry.draw_contour(draw_refine_img, color=label.obj_class.color, config=label.obj_class.geometry_config, thickness=3)

            draw_detect_img = ((draw_detect_img.astype(np.uint16) + raw_img.astype(np.uint16)) / 2).astype(np.uint8)
            draw_refine_img = ((draw_refine_img.astype(np.uint16) + raw_img.astype(np.uint16)) / 2).astype(np.uint8)

            # Get textline ground truths
            gt_objs = []
            for idx, label in reversed(list(enumerate(ann.labels))):
                if label.obj_class.name != TEXTLINE.lower():
                    crop_img = raw_img[label.geometry.top:label.geometry.bottom, label.geometry.left:label.geometry.right]
                    if crop_img.size == 0:
                        continue

                    crop_box, ret_ = ip.get_binary_area_coordinates_by_threshold(crop_img, min_thresh=127,
                                                                                 max_thresh=255)
                    if ret_:
                        crop_box_obj = ic.Box(crop_box)
                        proc_box = ic.calc_global_box_pos_in_box(g_box=[label.geometry.left, label.geometry.right, label.geometry.top, label.geometry.bottom],
                                                                 box=crop_box_obj.rect2,
                                                                 format='rect2')
                        min_x, max_x, min_y, max_y = proc_box
                        # debug_img = raw_img[min_y:max_y, min_x:max_x]
                        # print('Prev geo. : ', ann.labels[idx].geometry.left, ann.labels[idx].geometry.right, ann.labels[idx].geometry.top, ann.labels[idx].geometry.bottom)

                        # Remove label
                        ann = ann.delete_label(label)

                        # update coordinates
                        crop_labels = label.crop(sly.Rectangle(min_y, min_x, max_y, max_x))  # top, left, bottom, right,
                        for crop_label in crop_labels:
                            ann = ann.add_label(crop_label)
                        # print('Next geo. : ', ann.labels[idx].geometry.left, ann.labels[idx].geometry.right, ann.labels[idx].geometry.top, ann.labels[idx].geometry.bottom)

                    continue

                # Remove textline object
                ann = ann.delete_label(label)

                if (label.geometry.right - label.geometry.left) <= 0 or (label.geometry.bottom - label.geometry.top) <= 0:
                    continue

                gt_box = ic.Box(box=[[label.geometry.left, label.geometry.top], [label.geometry.right, label.geometry.bottom]])
                gt_obj = object.Object(name=TEXTLINE.lower(), box_obj=gt_box, description=label.description.strip())
                gt_objs.append(gt_obj)

            # Get predict results
            pred_objs = []
            for detector in [ko_detector, math_detector]:
                tgt_class = KO if (detector is ko_detector) else (MATH if (detector is math_detector) else 'None')

                # # Make border
                border_margin = 0
                if border_:
                    border_color = ig.WHITE
                    border_margin = 30
                    raw_img = cv2.copyMakeBorder(raw_img,
                                                   border_margin, border_margin, border_margin, border_margin,
                                                   cv2.BORDER_CONSTANT, value=border_color)

                boxes = get_textbox(detector, raw_img,
                                    canvas_size=craft_params['canvas_size'], mag_ratio=craft_params['mag_ratio'],
                                    text_threshold=craft_params['text_threshold'],
                                    link_threshold=craft_params['link_threshold'],
                                    low_text=craft_params['low_text'], poly=False,
                                    device=device, optimal_num_chars=True)

                if border_:
                    boxes = [np.array([box[0] - border_margin, box[1] - border_margin, box[2] - border_margin,
                                       box[3] - border_margin,
                                       box[4] - border_margin, box[5] - border_margin, box[6] - border_margin,
                                       box[7] - border_margin]) for box in boxes]

                horizontal_list, _ = group_text_box(boxes, craft_params['slope_ths'],
                                                    craft_params['ycenter_ths'], craft_params['height_ths'],
                                                    craft_params['width_ths'], craft_params['add_margin'])

                for h_box in horizontal_list:
                    pred_box = ic.Box(box=[[h_box[0], h_box[2]], [h_box[1], h_box[3]]])
                    pred_obj = object.Object(name=tgt_class.lower(), box_obj=pred_box, description='')
                    pred_objs.append(pred_obj)

                    if tgt_class == KO:
                        box_color = ig.BROWN
                    if tgt_class == MATH:
                        box_color = ig.MAGENTA

                    draw_detect_img = ip.draw_box_on_img(draw_detect_img, pred_box.flat_box, color=box_color, thickness=2)

            # Save result image
            ko_model_epoch, math_model_epoch = vars['ko_model_name'].split('_')[-1].replace('.pth', ''), \
                                               vars['math_model_name'].split('_')[-1].replace('.pth', '')
            rst_dir_name = f'{KO.lower()}_' + ko_model_epoch + '_' + f'{MATH.lower()}_' + math_model_epoch
            rst_dir_path = os.path.join(vars['rst_path'], rst_dir_name, 'draw_box')
            if save_detect_box_img_:
                cg.folder_exists(rst_dir_path, create_=True)
                ig.imwrite(draw_detect_img, os.path.join(rst_dir_path, f'[{img_mode}] ' + item_name))

            # Compare GT. & PRED.
            refine_gts = refine_ground_truths_by_predict_values(gt_objs, pred_objs, raw_img)  # test input : GTS, PREDS

            # Draw refined boxes & texts
            if save_refine_box_img_:
                for (rf_box, rf_text, rf_class) in refine_gts:
                    rf_rect2 = ic.convert_rect4_to_rect2(rf_box)
                    x_min, x_max, y_min, y_max = rf_rect2
                    box = (x_min, y_min, x_max, y_max)

                    if rf_class == KO.lower():
                        box_color = ig.BROWN
                    if rf_class == MATH.lower():
                        box_color = ig.MAGENTA

                    # Draw boxes
                    draw_refine_img = ip.draw_box_on_img(draw_refine_img, box, color=box_color, thickness=3)

                    # Draw texts (for 한글)
                    pil_img = Image.fromarray(draw_refine_img)
                    draw = ImageDraw.Draw(pil_img)
                    font = cg.KOR_FONT
                    margin_x, margin_y = 10, 45
                    draw.text(xy=((x_min + 1 + margin_x), (y_min + 1 + margin_y)),
                              text=rf_text, font=font, fill=box_color)

                    draw_refine_img = np.array(pil_img)

                rst_dir_path = os.path.join(vars['rst_path'], rst_dir_name, 'refine_box')
                cg.folder_exists(rst_dir_path, create_=True)
                ig.imwrite(draw_refine_img, os.path.join(rst_dir_path, f'[{img_mode}] ' + item_name))

            # Insert refine_gts to json
            if len(ann.labels) > 0:
                obj_id = ann.labels[-1].geometry.sly_id + 1
            else:
                obj_id = 0
            refine_json_data, refine_obj_id = update_json_from_results(ann.to_json(), obj_id,
                                                                       [KO.lower(), MATH.lower()], refine_gts)

            # Save refined json
            rst_ann_fname = item_paths.ann_path.replace(vars['textline_dataset_path'], vars['refine_dataset_path'])
            with open(rst_ann_fname, 'w', encoding='utf-8') as f:
                json.dump(refine_json_data, f, ensure_ascii=False, indent=4)

            sly.logger.info('[{}/{}] Refined json path : {}'.format(item_idx + 1, len(dataset), rst_ann_fname))

    logger.info(" # {} in {} mode finished.".format(_this_basename_, OP_MODE))
    return True


def link_or_copy_datasets(src_dir_path, dst_dir_path, dir_names, except_dir_names=None, tgt_dir_name='img/', mode=LINK, logger=None):
    if dir_names:
        for dir_name in dir_names:
            if dir_name in except_dir_names:
                logger.info(" # {} has already been split. ".format(dir_name))
                continue

            src_path = os.path.join(src_dir_path, dir_name, tgt_dir_name)

            dst_path = os.path.join(dst_dir_path, dir_name, tgt_dir_name)

            if cg.folder_exists(dst_path):
                logger.info(" # Already {} is exist".format(dst_path))
            else:
                cg.folder_exists(dst_path, create_=True)

            # check & link img_path, ann_path
            extensions = ig.IMG_EXTENSIONS if mode == LINK else jg.META_EXTENSION
            src_fnames = sorted(cg.get_filenames(src_path, extensions=extensions))
            src_bnames = [os.path.basename(src_fname) for src_fname in src_fnames]
            dst_fnames = sorted(cg.get_filenames(dst_path, extensions=extensions))
            dst_bnames = [os.path.basename(dst_fname) for dst_fname in dst_fnames]

            if any(src_bname not in dst_bnames for src_bname in src_bnames):
                if mode == LINK:
                    sym_cmd = 'ln "{}"* "{}"'.format(src_path, dst_path)  # to all files
                    subprocess.call(sym_cmd, shell=True)
                elif mode == COPY:
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

                logger.info(" # {} {} files {}\n{}->\t{}.".format(mode, tgt_dir_name.replace('/', ''), src_path, MARGIN, dst_path))
    else:
        logger.info(" [SPLIT-TEXTLINE] # Sorted dataset is empty !!!")


def refine_ground_truths_by_predict_values(gt_objs, pred_objs, img):
    refine_gts = []
    for gt_idx, gt_obj in enumerate(gt_objs):
        gt_box = gt_obj.Box

        # 중심점으로 gt 내부에 있는 pred. 후보 영역 추출
        cand_objs = []
        for pred_obj in pred_objs:
            pred_box = pred_obj.Box
            pred_center_x, pred_center_y = (pred_box.x1 + pred_box.x2) / 2, (pred_box.y1 + pred_box.y2) / 2

            if (gt_box.x1 < pred_center_x < gt_box.x2) and (gt_box.y1 < pred_center_y < gt_box.y2):
                cand_objs.append(pred_obj)

        # (x, y) 좌표를 기반으로 sorting
        sort_objs = sorted(cand_objs, key=lambda x: (x.Box.box[0][0], x.Box.box[0][1]))

        # 박스 좌표 및 사이즈로 중복 or 포함된 preds. 박스 제거
        for i, sort_obj in reversed(list(enumerate(sort_objs))):
            sort_box = sort_obj.Box
            _, sort_center_y = (sort_box.x1 + sort_box.x2) / 2, (sort_box.y1 + sort_box.y2) / 2
            sort_box_size = (sort_box.x2 - sort_box.x1) * (sort_box.y2 - sort_box.y1)
            if len(sort_objs) > 1:
                for j, ref_obj in reversed(list(enumerate(sort_objs[:i]))):
                    ref_box = ref_obj.Box
                    ref_box_size = (ref_box.x2 - ref_box.x1) * (ref_box.y2 - ref_box.y1)

                    # 두 박스의 넓이가 90% 이상 일치하거나 박스 4점이 모두 포함되면 제거
                    if ((sort_box_size / ref_box_size) >= 0.9) and (abs(sort_box.x1 - ref_box.x1) <= 10) or \
                            ((ref_box.x1 < sort_box.x1 < ref_box.x2) and (ref_box.x1 < sort_box.x2 < ref_box.x2) and
                             (ref_box.y1 < sort_box.y1 < ref_box.y2) and (ref_box.y1 < sort_box.y2 < ref_box.y2)):
                        del sort_objs[i]
                        break

        remove_objs = sort_objs

        split_gts = []
        # 예측 값이 없는 경우
        if len(remove_objs) == 0:
            split_gts.append([gt_box.rect4, '', MATH.lower()])
        else:
            # Create refined gts
            for k, remove_obj in enumerate(remove_objs):
                remove_class, remove_box = remove_obj.name, remove_obj.Box

                # 예측 개수를 기반으로 x, y값 조정
                if len(remove_objs) == 1:
                    split_gts.append([gt_box.rect4, '', remove_class])

                # 예측 개수가 2개 이상 일때
                else:
                    # 첫번째 영역 처리
                    if k == 0:
                        rect4 = ic.convert_rect2_to_rect4([gt_box.x1, remove_box.x2, gt_box.y1, gt_box.y2])
                        split_gts.append([rect4, '', remove_class])

                    # 마지막 영역 처리
                    elif k == len(remove_objs) - 1:
                        rect4 = ic.convert_rect2_to_rect4([remove_box.x1, gt_box.x2, gt_box.y1, gt_box.y2])
                        split_gts.append([rect4, '', remove_class])

                    # 중간 영역 처리
                    else:
                        rect4 = ic.convert_rect2_to_rect4([remove_box.x1, remove_box.x2, gt_box.y1, gt_box.y2])
                        split_gts.append([rect4, '', remove_class])

        # 이미지 처리를 통해 박스 좌표 교정
        proc_gts = []
        for i, split_gt in enumerate(split_gts):
            split_gt_box, split_gt_text, split_gt_class = split_gt
            [g_min_x, g_max_x, g_min_y, g_max_y] = ic.convert_rect4_to_rect2(split_gt_box)
            crop_img = img[g_min_y:g_max_y, g_min_x:g_max_x]

            crop_box, ret_ = ip.get_binary_area_coordinates_by_threshold(crop_img, min_thresh=127, max_thresh=255)
            if ret_:
                crop_box_obj = ic.Box(crop_box)
                proc_box = ic.calc_global_box_pos_in_box(g_box=[g_min_x, g_max_x, g_min_y, g_max_y],
                                                         box=crop_box_obj.rect2,
                                                         format='rect2')

                proc_gts.append([ic.convert_rect2_to_rect4(proc_box), split_gt_text, split_gt_class])

        # pred_class를 기반으로 text filling
        ch_pos = 0
        for l, proc_gt in enumerate(proc_gts):
            proc_gt_box, proc_gt_text, proc_gt_class = proc_gt
            refine_gts.append([proc_gt_box, '', proc_gt_class])

            gt_text = gt_obj.description
            for m, (prev_ch, curr_ch, next_ch) in enumerate(cs.get_prev_and_next(gt_text[ch_pos:])):
                if (len(gt_text) <= 1) and (curr_ch == ' ' or curr_ch == ''):
                    refine_gts[-1][1] += curr_ch
                    ch_pos += 1
                    break

                # 첫번째 문자 처리
                if ch_pos == 0:
                    # 한글 이면
                    if cs.is_korean(curr_ch):
                        refine_gts[-1][1] += curr_ch
                        ch_pos += 1

                    # 수식 이면
                    elif not cs.is_korean(curr_ch):
                        refine_gts[-1][1] += curr_ch
                        ch_pos += 1

                else:
                    # (한글+빈칸) and (prev_ch_class == curr_ch_class)
                    if (cs.is_korean(prev_ch) and (curr_ch == ' ')) or (
                            (prev_ch == ' ' or prev_ch is None) and cs.is_korean(curr_ch)) \
                            or (cs.is_korean(prev_ch) and cs.is_korean(curr_ch)):
                        refine_gts[-1][1] += curr_ch
                        ch_pos += 1

                    # (수식+빈칸) and (prev_ch_class == curr_ch_class)
                    elif (not (cs.is_korean(prev_ch)) and (curr_ch == ' ')) or (
                            (prev_ch == ' ' or prev_ch is None) and not (cs.is_korean(curr_ch))) \
                            or ((cs.is_korean(prev_ch) == False) and (cs.is_korean(curr_ch) == False)):
                        refine_gts[-1][1] += curr_ch
                        ch_pos += 1

                    # class가 바뀔때
                    curr_class = refine_gts[-1][2]
                    if (curr_class == KO.lower() and not (cs.is_korean(next_ch)) and (next_ch != ' ')) or \
                            (curr_class == MATH.lower() and (cs.is_korean(next_ch)) and (next_ch != ' ')) or \
                            next_ch == None:
                        break

    return refine_gts


def update_json_from_results(json_data, obj_id, class_names, results):
    for i, (box, value, class_name) in enumerate(results):
        if class_name in class_names:
            rect2 = ic.convert_rect4_to_rect2(box)
            update_obj = get_obj_data(obj_id, class_name, rect2, value)
            json_data['objects'].append(update_obj)
            obj_id += 1

    return json_data, obj_id


def get_obj_data(obj_id, classTitle, box, value):
    if classTitle == TABLE.lower():
        classId = 2790491
        value = ''
    elif classTitle == GRAPH.lower():
        classId = 2772037
        value = ''
    elif classTitle == TEXTLINE.lower():
        classId = 2772036
    elif classTitle == MATH.lower():
        classId = 2883527
    elif classTitle == KO.lower():
        classId = 2883530

    date = datetime.today().strftime("%Y%m%d%H%M%S")
    year, month, day, hour, minute, second = date[:4], date[4:6], date[6:8], date[8:10], date[10:12], date[12:14]

    obj_data = {}
    update_obj = update_obj_data(obj_data,
                                 id=obj_id, classId=classId,
                                 description=value, geometryType='rectangle',
                                 labelerLogin='freewheelin',
                                 createdAt=f'{year}-{month}-{day}T{hour}:{minute}:{second}.271Z',
                                 updatedAt=f'{year}-{month}-{day}T{hour}:{minute}:{second}.271Z',
                                 classTitle=classTitle,
                                 tags=[],
                                 points={
                                     'exterior': [[int(box[0]), int(box[2])],
                                                  [int(box[1]), int(box[3])]],
                                     'interior': [[]],
                                 })
    return update_obj


def update_obj_data(obj_data, id, classId, description, geometryType, labelerLogin, createdAt, updatedAt, tags,
                    classTitle, points):
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
    ini = cg.get_ini_parameters(args.ini_fname)
    common_info = {}
    for key, val in ini['COMMON'].items():
        common_info[key] = val

    logger = cl.setup_logger_with_ini(ini['LOGGER'],
                                      logging_=args.logging_, console_=args.console_logging_)

    if args.op_mode == PREPROCESS_ALL:
        # Preprocess ko, math dataset
        dataset_types = KO_MATH.split('_')
        for dataset_type in dataset_types:
            # Reload ini
            ini = cg.get_ini_parameters(f'craft_learn_{dataset_type.lower()}.ini')
            common_info = {}
            for key, val in ini['COMMON'].items():
                common_info[key] = val

            # Init. local variables
            vars = {}
            for key, val in ini[PREPROCESS_ALL].items():
                vars[key] = cs.replace_string_from_dict(val, common_info)

            # Run generate & split
            tgt_dir_names = vars['tgt_dir_names'].replace(' ', '').split(',')
            for tgt_dir_name in tgt_dir_names:
                common_info['tgt_dir_name'] = tgt_dir_name
                main_generate(ini[GENERATE], common_info, logger=logger)
                main_split(ini[SPLIT], common_info, logger=logger)

            # Run merge
            main_merge(ini[MERGE], common_info, logger=logger)

    elif args.op_mode == GENERATE:
        main_generate(ini[GENERATE], common_info, logger=logger)
    elif args.op_mode == SPLIT:
        main_split(ini[SPLIT], common_info, logger=logger)
    elif args.op_mode == MERGE:
        main_merge(ini[MERGE], common_info, logger=logger)
    elif args.op_mode == TRAIN:
        main_train(ini[TRAIN], common_info, logger=logger)
    elif args.op_mode == TEST:
        main_test(ini[TEST], common_info, logger=logger)
    elif args.op_mode == TRAIN_TEST:
        ret, model_dir = main_train(ini[TRAIN], common_info, logger=logger)
        main_test(ini[TEST], common_info, logger=logger)
        print(" # Trained model directory is {}".format(model_dir))
    elif args.op_mode == SPLIT_TEXTLINE:
        main_split_textline(ini[SPLIT_TEXTLINE], common_info, logger=logger)
    else:
        print(" @ Error: op_mode, {}, is incorrect.".format(args.op_mode))

    return True


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", required=True, choices=[TEXTLINE, KO, MATH, KO_MATH], help="dataset type")
    parser.add_argument("--op_mode", required=True, choices=[PREPROCESS_ALL, GENERATE, MERGE, SPLIT, TRAIN, TEST, TRAIN_TEST, SPLIT_TEXTLINE], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--model_dir", default="", help="Model directory")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
DATASET_TYPE = TEXTLINE  # KO / MATH / KO_MATH / TEXTLINE
OP_MODE = SPLIT_TEXTLINE
# PREPROCESS_ALL
# (GENERATE / SPLIT / MERGE)
# TRAIN / TEST / TRAIN_TEST / SPLIT_TEXTLINE

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
