import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict

import cv2
import numpy as np
from easyocr.craft_utils import getDetBoxes, adjustResultCoordinates
from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
from easyocr.craft import CRAFT

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

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, canvas_size,\
                                                                          interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, polys, mapper = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
    if estimate_num_chars:
        boxes = list(boxes)
        polys = list(polys)
    for k in range(len(polys)):
        if estimate_num_chars:
            boxes[k] = (boxes[k], mapper[k])
        if polys[k] is None: polys[k] = boxes[k]

    return boxes, polys

def get_detector(trained_model, device='cpu', quantize=True):
    net = CRAFT(pretrained=False)

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = False

    net.eval()
    return net

def get_textbox(detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device, optimal_num_chars=None):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes, polys = test_net(canvas_size, mag_ratio, detector, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars)

    if estimate_num_chars:
        polys = [p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]

    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    return result
