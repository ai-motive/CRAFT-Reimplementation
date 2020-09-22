import os
import sys
import json
import pprint
import time
import argparse
import general_utils as utils
import file_utils
import coordinates as coord


_this_folder_ = os.path.dirname(os.path.abspath(__file__))
_this_basename_ = os.path.splitext(os.path.basename(__file__))[0]


def main(args):
    ini = utils.get_ini_parameters(args.ini_fname)
    logger = utils.setup_logger_with_ini(ini['LOGGER'],
                                         logging_=args.logging_, console_=args.console_logging_)
    if args.op_mode == 'convert_ic15':
        utils.folder_exists(args.out_path, create_=True)
        if os.path.isdir(args.in_path):
            utils.copy_folder_structure(args.in_path, args.out_path)

        img_path = os.path.join(args.in_path, 'img/')
        ann_path = os.path.join(args.in_path, 'ann/')
        gt_path = os.path.join(args.out_path, 'gt/')
        img_fnames = utils.get_filenames(img_path, extensions=utils.IMG_EXTENSIONS)
        ann_fnames = utils.get_filenames(ann_path, extensions=utils.META_EXTENSION)
        logger.info(" [PREPROCESS] # Total file number to be processed: {:d}.".format(len(img_fnames)))

        for idx, fname in enumerate(ann_fnames):
            logger.info(" [SYS-OCR] # Processing {} ({:d}/{:d})".format(fname, (idx + 1), len(img_fnames)))

            dir_name, core_name, ext = utils.split_fname(fname)
            rst_path = gt_path

            img = utils.imread(img_path + core_name, color_fmt='RGB')

            # Load json
            with open(fname) as json_file:
                json_data = json.load(json_file)
                objects = json_data['objects']
                # pprint.pprint(objects)

            bboxes = []
            for obj in objects:
                class_name = obj['classTitle']
                if class_name in ['problem_whole', 'ignore', 'graph_diagrams']:
                    continue

                [x1, y1], [x2, y2] = obj['points']['exterior']
                x_min, y_min, x_max, y_max = int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))
                if x_max - x_min <= 0 or y_max - y_min <= 0:
                    continue

                rect4 = coord.convert_rect2_to_rect4([x_min, x_max, y_min, y_max])
                bboxes.append(rect4)

            file_utils.saveResult(img_file=core_name, img=img, boxes=bboxes, dirname=rst_path)

        logger.info(" # {} in {} mode finished.".format(_this_basename_, args.op_mode))
    pass

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--op_mode", required=True, choices=['convert_ic15',], help="operation mode")
    parser.add_argument("--ini_fname", required=True, help="System code ini filename")
    parser.add_argument("--in_path", required=True, type=str, help="input file")
    parser.add_argument("--out_path", default=".", help="Output folder")

    parser.add_argument("--logging_", default=False, action='store_true', help="Activate logging")
    parser.add_argument("--console_logging_", default=False, action='store_true', help="Activate logging")

    args = parser.parse_args(argv)

    return args


SELF_TEST_ = True
OP_MODE = 'convert_ic15'
INI_FNAME = _this_basename_ + ".ini"
# DEBUG_PATH = "../Debug/IMGs/쎈_수학(상)2/"
IN_PATH = "../data/CRAFT-pytorch/Light_SSen(top)/"
OUT_PATH = "../output/CRAFT-pytorch/Light_SSen(top)/"


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